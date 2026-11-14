# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang_rbln.compilation import backend, device_views

BASE = (1, 8, 4, 64)


def base_tensor() -> torch.Tensor:
    return torch.randn(BASE, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize(
    ("make", "slices"),
    [
        (lambda b: b[:, :, 1, :], (("select", 2, 1),)),
        (lambda b: b[:, :, 3], (("select", 2, 3),)),
        (lambda b: b[:, 2:6], (("narrow", 1, 2, 4),)),
        (lambda b: b[:, 2:6, 1], (("narrow", 1, 2, 4), ("select", 2, 1))),
        (
            lambda b: b[0, :, 2, 16:48],
            (("select", 0, 0), ("select", 1, 2), ("narrow", 1, 16, 32)),
        ),
    ],
    ids=[
        "select",
        "select-last-kept",
        "narrow",
        "narrow+select",
        "select+select+narrow",
    ],
)
def test_finds_the_slices_of_a_sliced_view(make, slices):
    base = base_tensor()
    view = make(base)
    assert device_views.slices_of(view) == slices
    torch.testing.assert_close(
        device_views.apply_slices(base, slices), view, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda b: b.transpose(1, 2),
        lambda b: b[:, :, :, ::2],
        lambda b: b.contiguous()[:, ::2],
    ],
    ids=["transpose", "step-2", "strided-step"],
)
def test_leaves_views_slicing_cannot_rebuild(make):
    assert device_views.slices_of(make(base_tensor())) is None


def layer(x, per_layer, other):
    return x * 2 + per_layer + other.sum()


def compile_with_rebase(fn, seen):
    def stand_in(graph_module, example_inputs):
        rebased = device_views.rebase_strided_views(
            graph_module, example_inputs, lambda value: True
        )
        seen.append((rebased, [tuple(v.shape) for v in example_inputs]))

        def run(*args):
            if rebased is not None:
                args = rebased.bases(args)
            return graph_module(*args)

        return run

    return torch.compile(fn, backend=stand_in, dynamic=False)


def test_rewritten_graph_takes_the_base_and_matches_eager():
    torch._dynamo.reset()
    seen = []
    compiled = compile_with_rebase(layer, seen)
    for step in range(3):
        x = torch.randn(1, 8, 64, generator=torch.Generator().manual_seed(step))
        per_layer_inputs = torch.randn(
            1, 8, 4, 64, generator=torch.Generator().manual_seed(10 + step)
        )
        other = torch.randn(3)
        view = per_layer_inputs[:, :, 2, :]
        torch.testing.assert_close(
            compiled(x, view, other), layer(x, view, other), rtol=0, atol=0
        )
    ((rebased, _),) = seen
    assert rebased is not None and rebased.slices == {1: (("select", 2, 2),)}


def test_two_views_of_one_base_each_get_their_slice():
    torch._dynamo.reset()

    def two(a, b):
        return a - b

    seen = []
    compiled = compile_with_rebase(two, seen)
    base = base_tensor()
    torch.testing.assert_close(
        compiled(base[:, :, 0], base[:, :, 3]),
        base[:, :, 0] - base[:, :, 3],
        rtol=0,
        atol=0,
    )
    ((rebased, _),) = seen
    assert rebased.slices == {0: (("select", 2, 0),), 1: (("select", 2, 3),)}


def test_a_changed_view_is_refused_not_misread():
    rebased = device_views.RebasedViews({0: (("select", 2, 1),)}, {0: BASE})
    base = base_tensor()
    rebased.bases([base[:, :, 1]])
    with pytest.raises(RuntimeError, match="no longer the same view"):
        rebased.bases([base[:, :, 2]])


def test_backend_hands_the_compiled_program_the_base(monkeypatch):
    torch._dynamo.reset()
    monkeypatch.setattr(backend, "on_device", lambda value: True)
    calls = []

    def program_backend(graph_module, example_inputs):
        calls.append(("compile", [v.is_contiguous() for v in example_inputs]))

        def run(*args):
            calls.append(("run", [tuple(a.shape) for a in args]))
            return graph_module(*args)

        return run

    compiled = torch.compile(
        layer, backend=backend._instrumented_backend(program_backend), dynamic=False
    )
    x, per_layer_inputs, other = torch.randn(1, 8, 64), base_tensor(), torch.randn(3)
    view = per_layer_inputs[:, :, 1, :]
    torch.testing.assert_close(
        compiled(x, view, other), layer(x, view, other), rtol=0, atol=0
    )
    assert calls[0] == ("compile", [True, True, True])
    assert calls[1] == ("run", [(1, 8, 64), BASE, (3,)])


def test_inference_mode_views_slice_their_storage():
    # inference_mode records no _base: the base comes from the view's storage
    with torch.inference_mode():
        projection = torch.randn(1, 8, 4 * 64).reshape(1, 8, 4, 64)
        view = projection[:, :, 2, :]
        assert view._base is None
        slices = device_views.slices_of(view)
        base = device_views.storage_base(view)
    assert slices == (("narrow", 2, 128, 64),)
    assert tuple(base.shape) == (1, 8, 256)
    torch.testing.assert_close(
        device_views.apply_slices(base, slices), view, rtol=0, atol=0
    )


def test_inference_mode_graph_takes_the_storage_base():
    torch._dynamo.reset()
    seen = []
    compiled = compile_with_rebase(layer, seen)
    with torch.inference_mode():
        for step in range(2):
            x = torch.randn(1, 8, 64, generator=torch.Generator().manual_seed(step))
            per_layer_inputs = torch.randn(
                1, 8, 256, generator=torch.Generator().manual_seed(5 + step)
            ).reshape(1, 8, 4, 64)
            other = torch.randn(3)
            view = per_layer_inputs[:, :, 1, :]
            torch.testing.assert_close(
                compiled(x, view, other), layer(x, view, other), rtol=0, atol=0
            )
    ((rebased, shapes),) = seen
    assert rebased is not None and rebased.slices == {1: (("narrow", 2, 64, 64),)}
    assert shapes[1] == (1, 8, 64)


def test_an_offset_row_slice_is_rebased_though_contiguous():
    # hidden[:, n - 1 : n]: contiguous by its size-1 dims, but its address is inside
    # another tensor's buffer
    torch._dynamo.reset()
    seen = []

    def head(row):
        return row * 3

    compiled = compile_with_rebase(head, seen)
    with torch.inference_mode():
        hidden = torch.randn(1, 8, 64)
        row = hidden[:, 6:7]
        assert row.is_contiguous() and row.storage_offset() != 0
        torch.testing.assert_close(compiled(row), row * 3, rtol=0, atol=0)
    ((rebased, _),) = seen
    assert rebased is not None and rebased.slices == {0: (("narrow", 1, 6, 1),)}


def test_inputs_sharing_a_storage_reach_the_graph_in_one_shape():
    # the decode step of Gemma-4: layer 0's per-layer slice starts the storage
    # (contiguous, offset 0) while the other layers' slices are offset into it
    torch._dynamo.reset()
    seen = []

    def two(first, second):
        return first + second

    compiled = compile_with_rebase(two, seen)
    with torch.inference_mode():
        per_layer_inputs = torch.randn(1, 1, 3 * 64).reshape(1, 1, 3, 64)
        first, second = per_layer_inputs[:, :, 0, :], per_layer_inputs[:, :, 2, :]
        assert first.is_contiguous() and first.storage_offset() == 0
        torch.testing.assert_close(
            compiled(first, second), first + second, rtol=0, atol=0
        )
    ((rebased, _),) = seen
    assert rebased.base_shapes == {0: (1, 1, 192), 1: (1, 1, 192)}
    assert rebased.slices == {0: (("narrow", 2, 0, 64),), 1: (("narrow", 2, 128, 64),)}
