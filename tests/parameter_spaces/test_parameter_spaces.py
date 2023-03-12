import pytest
import torch

from opof.parameter_spaces import Interval, Simplex, Sphere


def test_interval_invalid():
    # Invalid count.
    with pytest.raises(ValueError):
        Interval(0)
    with pytest.raises(ValueError):
        Interval(-5)


def test_interval_rand():
    for c in [1, 10, 100]:
        space = Interval(c)

        # Range over batch size.
        for b in [1, 10]:
            p = space.rand(b)

            # Check parameters.
            assert p.shape == torch.Size([b, c, 1])


def test_interval_trans():
    # Range over number of values.
    for c in [1, 10, 100]:
        space = Interval(c)
        assert space.trans_num_inputs == c

        # Range over batch size.
        for b in [1, 10]:
            trans_inputs = torch.rand(b, space.trans_num_inputs)
            (p, _) = space.trans_forward(trans_inputs)

            # Check parameters.
            assert p.shape == torch.Size([b, c, 1])


def test_interval_dist():
    # Range over number of values.
    for c in [1, 10, 100]:
        space = Interval(c)
        assert len(space.dist_target_entropy) == c
        sampler = space.create_sampler()

        # Range over batch size.
        for b in [1, 10]:
            dist_inputs = torch.randn((b, space.dist_num_inputs))
            (p, e, _) = sampler(dist_inputs)

            # Check parameters.
            assert p.shape[0] == b  # Check batch size.
            assert p.shape[1] == c  # Check count.
            assert p.shape[2] == 1  # Check value.

            # Check entropy.
            assert len(e.shape) == 2  # Entropy terms should be flattened.
            assert e.shape[0] == b
            assert e.shape[1] == c


def test_sphere_invalid():
    # Invalid count.
    with pytest.raises(ValueError):
        Sphere(0, 2)
    with pytest.raises(ValueError):
        Sphere(-1, 2)


def test_sphere_rand():
    # Range over number of values.
    for c in [1, 10, 100]:
        # Range over dimension.
        for d in [2, 5, 20]:
            space = Sphere(c, d)

            # Range over batch size.
            for b in [1, 10]:
                p = space.rand(b)

                # Check parameters.
                assert p.shape == torch.Size([b, c, d])


def test_sphere_trans():
    # Range over number of values.
    for c in [1, 10, 100]:
        # Range over dimension.
        for d in [2, 5, 20]:
            space = Sphere(c, d)
            assert space.trans_num_inputs == c * d

            # Range over batch size.
            for b in [1, 10]:
                trans_inputs = torch.rand(b, space.trans_num_inputs)
                (p, _) = space.trans_forward(trans_inputs)

                # Check parameters.
                assert p.shape == torch.Size([b, c, d])


def test_sphere_dist():
    # Range over number of values.
    for c in [1, 10, 100]:
        # Range over dimension.
        for d in [2, 5, 20]:
            space = Sphere(c, d)
            assert len(space.dist_target_entropy) == c
            sampler = space.create_sampler()

            # Range over batch size.
            for b in [1, 10]:
                dist_inputs = torch.randn((b, space.dist_num_inputs))
                (p, e, _) = sampler(dist_inputs)

                # Check parameters.
                assert p.shape[0] == b  # Check batch size.
                assert p.shape[1] == c  # Check count.
                assert p.shape[2] == d  # Check value.

                # Check entropy.
                assert len(e.shape) == 2  # Entropy terms should be flattened.
                assert e.shape[0] == b
                assert e.shape[1] == c


def test_simplex_invalid():
    # Invalid count.
    with pytest.raises(ValueError):
        Simplex(0, 2)
    with pytest.raises(ValueError):
        Simplex(-1, 2)


def test_simplex_rand():
    # Range over number of values.
    for c in [1, 10, 100]:
        # Range over dimension.
        for choices in [2, 5, 20]:
            space = Simplex(c, choices)

            # Range over batch size.
            for b in [1, 10]:
                p = space.rand(b)

                # Check parameters.
                assert p.shape == torch.Size([b, c, choices])


def test_simplex_trans():
    # Range over number of values.
    for c in [1, 10, 100]:
        # Range over dimension.
        for choices in [2, 5, 20]:
            space = Simplex(c, choices)
            assert len(space.dist_target_entropy) == c

            # Range over batch size.
            for b in [1, 10]:
                trans_inputs = torch.rand(b, space.trans_num_inputs)
                (p, _) = space.trans_forward(trans_inputs)

                # Check parameters.
                assert p.shape == torch.Size([b, c, choices])


def test_simplex_dist():
    # Range over number of values.
    for c in [1, 10, 100]:
        # Range over dimension.
        for choices in [2, 5, 20]:
            space = Simplex(c, choices)
            assert len(space.dist_target_entropy) == c
            sampler = space.create_sampler()

            # Range over batch size.
            for b in [1, 10]:
                dist_inputs = torch.randn((b, space.dist_num_inputs))
                (p, e, _) = sampler(dist_inputs)

                # Check parameters.
                assert p.shape[0] == b  # Check batch size.
                assert p.shape[1] == c  # Check count.
                assert p.shape[2] == choices  # Check value.

                # Check entropy.
                assert len(e.shape) == 2  # Entropy terms should be flattened.
                assert e.shape[0] == b
                assert e.shape[1] == c
