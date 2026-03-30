"""Tests for cross_validate_lambda and _select_lambda_1se."""

import numpy as np
import pytest
import torch

from colabbtr.morphology import (
    _select_lambda_1se,
    cross_validate_lambda,
    differentiable_btr,
    idilation,
    ierosion,
)


class TestSelectLambda1SE:
    """Unit tests for the one-standard-error rule selection logic."""

    def test_basic_u_shape(self):
        """Minimum in the middle, 1SE selects a larger lambda."""
        lambdas = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
        loss_mean = [10.0, 5.0, 3.0, 4.0, 8.0]  # min at index 2
        loss_std = [1.0, 1.0, 1.0, 1.0, 1.0]
        # threshold = 3.0 + 1.0 = 4.0
        # indices where loss <= 4.0: [1, 2, 3]
        # imin_1se = 3, loss_mean[3]=4.0 == threshold → frac=0
        result = _select_lambda_1se(lambdas, loss_mean, loss_std)
        assert result == pytest.approx(1.0)

    def test_minimum_at_last_index(self):
        """Minimum at last index — no interpolation possible, returns last lambda."""
        lambdas = np.array([0.001, 0.01, 0.1])
        loss_mean = [10.0, 5.0, 3.0]
        loss_std = [1.0, 1.0, 1.0]
        result = _select_lambda_1se(lambdas, loss_mean, loss_std)
        assert result == pytest.approx(0.1)

    def test_all_equal_loss(self):
        """All losses equal with zero std — selects the largest lambda."""
        lambdas = np.array([0.001, 0.01, 0.1])
        loss_mean = [5.0, 5.0, 5.0]
        loss_std = [0.0, 0.0, 0.0]
        result = _select_lambda_1se(lambdas, loss_mean, loss_std)
        assert result == pytest.approx(0.1)

    def test_single_lambda(self):
        """Single lambda — returns it directly."""
        lambdas = np.array([0.01])
        loss_mean = [5.0]
        loss_std = [1.0]
        result = _select_lambda_1se(lambdas, loss_mean, loss_std)
        assert result == pytest.approx(0.01)

    def test_interpolation(self):
        """Linear interpolation between grid points."""
        lambdas = np.array([1.0, 2.0, 3.0])
        loss_mean = [5.0, 3.0, 7.0]  # min at index 1
        loss_std = [1.0, 1.0, 1.0]
        # threshold = 3.0 + 1.0 = 4.0
        # candidates where <= 4.0: [1] only
        # imin_1se = 1, interpolate between index 1 (3.0) and 2 (7.0)
        # frac = (4.0 - 3.0) / (7.0 - 3.0) = 0.25
        # lambda_opt = 2.0 + (3.0 - 2.0) * 0.25 = 2.25
        result = _select_lambda_1se(lambdas, loss_mean, loss_std)
        assert result == pytest.approx(2.25)

    def test_minimum_at_first_index(self):
        """Minimum at first index, 1SE may still select larger lambda."""
        lambdas = np.array([0.01, 0.1, 1.0])
        loss_mean = [2.0, 2.5, 10.0]
        loss_std = [1.0, 1.0, 1.0]
        # threshold = 2.0 + 1.0 = 3.0
        # candidates: [0, 1] (2.0 and 2.5 both <= 3.0)
        # imin_1se = 1, interpolate between 1 (2.5) and 2 (10.0)
        # frac = (3.0 - 2.5) / (10.0 - 2.5) = 0.5/7.5
        expected = 0.1 + (1.0 - 0.1) * (0.5 / 7.5)
        result = _select_lambda_1se(lambdas, loss_mean, loss_std)
        assert result == pytest.approx(expected)


class TestCrossValidateLambda:
    """Integration tests using real data with minimal epochs."""

    @pytest.fixture(scope="module")
    def tiny_images(self):
        images_np = np.load("data/single_tip/images.npy")
        return torch.tensor(images_np[:5], dtype=torch.float64)

    def test_returns_correct_structure(self, tiny_images):
        lambda_opt, cv_result = cross_validate_lambda(
            tiny_images, tip_size=(5, 5),
            lambda_min=0.001, lambda_max=0.1, lambda_num=3,
            n_folds=5, nepoch=2, lr=0.1, is_tqdm=False,
        )
        assert isinstance(lambda_opt, float)
        assert lambda_opt > 0
        assert set(cv_result.keys()) == {'lambdas', 'loss_mean', 'loss_std', 'lambda_min_idx', 'lambda_1se_idx'}
        assert len(cv_result['lambdas']) == 3
        assert len(cv_result['loss_mean']) == 3
        assert len(cv_result['loss_std']) == 3
        assert isinstance(cv_result['lambda_min_idx'], int)
        assert isinstance(cv_result['lambda_1se_idx'], int)

    def test_lambda_in_range(self, tiny_images):
        lmin, lmax = 0.001, 0.1
        lambda_opt, _ = cross_validate_lambda(
            tiny_images, tip_size=(5, 5),
            lambda_min=lmin, lambda_max=lmax, lambda_num=3,
            n_folds=5, nepoch=2, lr=0.1, is_tqdm=False,
        )
        assert lmin <= lambda_opt <= lmax

    def test_too_few_images(self):
        images = torch.randn(3, 10, 10, dtype=torch.float64)
        with pytest.raises(ValueError, match="at least"):
            cross_validate_lambda(images, (5, 5), n_folds=5, nepoch=1, is_tqdm=False)

    def test_invalid_lambda_num(self):
        images = torch.randn(5, 10, 10, dtype=torch.float64)
        with pytest.raises(ValueError, match="lambda_num"):
            cross_validate_lambda(images, (5, 5), lambda_num=0, n_folds=5, nepoch=1, is_tqdm=False)

    def test_invalid_lambda_range(self):
        images = torch.randn(5, 10, 10, dtype=torch.float64)
        with pytest.raises(ValueError, match="positive"):
            cross_validate_lambda(images, (5, 5), lambda_min=-1, n_folds=5, nepoch=1, is_tqdm=False)
        with pytest.raises(ValueError, match="less than"):
            cross_validate_lambda(images, (5, 5), lambda_min=0.1, lambda_max=0.01, n_folds=5, nepoch=1, is_tqdm=False)


class TestNotebookEquivalence:
    """Verify cross_validate_lambda matches the notebook's inline implementation."""

    @pytest.fixture(scope="class")
    def shared_params(self):
        """Shared parameters and data for both implementations."""
        images_np = np.load("data/single_tip/images.npy")
        images = torch.tensor(images_np[:5], dtype=torch.float64)
        return dict(
            images=images,
            tip_height=5,
            tip_width=5,
            lambda_min=0.001,
            lambda_max=0.1,
            lambda_num=3,
            n_folds=5,
            epoch=3,
            learning_rate=0.1,
        )

    @pytest.fixture(scope="class")
    def notebook_result(self, shared_params):
        """Run the notebook's inline CV implementation (Cell 8)."""
        p = shared_params
        images = p["images"]
        n_samples = images.shape[0]
        lambdas = np.logspace(np.log10(p["lambda_min"]), np.log10(p["lambda_max"]), p["lambda_num"])

        # --- Exact copy of notebook's evaluate_loss_mean_and_std ---
        def evaluate_loss_mean_and_std(images_train, tip_height, tip_width, epoch, learning_rate, lambda1):
            n_folds = p["n_folds"]
            fold_size = n_samples // n_folds
            CV_loss = []

            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

                images_train = images[0:n_samples, :, :]

                train_indices = np.concatenate([np.arange(0, val_start), np.arange(val_end, n_samples)])

                train_data = torch.cat([images_train[i:i+1] for i in train_indices], dim=0)
                val_data = torch.cat([images_train[i:i+1] for i in range(val_start, val_end)], dim=0)

                tip, _ = differentiable_btr(train_data,
                                            (tip_height, tip_width),
                                            nepoch=epoch, lr=learning_rate, weight_decay=lambda1, is_tqdm=False)

                with torch.no_grad():
                    val_loss = 0
                    for i in range(val_data.shape[0]):
                        image = val_data[i, :, :]
                        image_reconstructed = idilation(ierosion(image, tip), tip)
                        loss_tmp = torch.mean((image_reconstructed - image)**2)
                        val_loss += loss_tmp.item()

                CV_loss.append(val_loss)

            return np.mean(CV_loss), np.std(CV_loss)

        # --- Exact copy of notebook's lambda loop ---
        loss_mean = []
        loss_std = []
        for lambda1 in lambdas:
            loss_mean_each, loss_std_each = evaluate_loss_mean_and_std(
                images, p["tip_height"], p["tip_width"], p["epoch"], p["learning_rate"], lambda1)
            loss_mean.append(loss_mean_each)
            loss_std.append(loss_std_each)

        # --- Exact copy of notebook's 1SE rule ---
        imin = np.argmin(loss_mean)
        loss_min_plus_std = loss_mean[imin] + loss_std[imin]
        imin_plus_std = np.where(loss_mean <= loss_min_plus_std)[0]
        imin_plus_std = np.max(imin_plus_std)

        # Linear interpolation (with guard for last-index edge case)
        if imin_plus_std + 1 < len(lambdas):
            lam_lo = lambdas[imin_plus_std]
            lam_hi = lambdas[imin_plus_std + 1]
            optimal_lambda = lam_lo + (lam_hi - lam_lo) * (
                loss_min_plus_std - loss_mean[imin_plus_std]
            ) / (loss_mean[imin_plus_std + 1] - loss_mean[imin_plus_std])
        else:
            optimal_lambda = lambdas[imin_plus_std]

        return dict(
            lambdas=lambdas,
            loss_mean=loss_mean,
            loss_std=loss_std,
            optimal_lambda=float(optimal_lambda),
        )

    @pytest.fixture(scope="class")
    def function_result(self, shared_params):
        """Run cross_validate_lambda with the same parameters."""
        p = shared_params
        optimal_lambda, cv_result = cross_validate_lambda(
            p["images"],
            tip_size=(p["tip_height"], p["tip_width"]),
            lambda_min=p["lambda_min"],
            lambda_max=p["lambda_max"],
            lambda_num=p["lambda_num"],
            n_folds=p["n_folds"],
            nepoch=p["epoch"],
            lr=p["learning_rate"],
            is_tqdm=False,
        )
        return dict(optimal_lambda=optimal_lambda, cv_result=cv_result)

    def test_lambdas_match(self, notebook_result, function_result):
        np.testing.assert_array_equal(
            notebook_result["lambdas"],
            function_result["cv_result"]["lambdas"],
        )

    def test_loss_mean_match(self, notebook_result, function_result):
        np.testing.assert_allclose(
            notebook_result["loss_mean"],
            function_result["cv_result"]["loss_mean"],
            rtol=1e-10,
        )

    def test_loss_std_match(self, notebook_result, function_result):
        np.testing.assert_allclose(
            notebook_result["loss_std"],
            function_result["cv_result"]["loss_std"],
            rtol=1e-10,
        )

    def test_optimal_lambda_match(self, notebook_result, function_result):
        assert function_result["optimal_lambda"] == pytest.approx(
            notebook_result["optimal_lambda"], rel=1e-10,
        )
