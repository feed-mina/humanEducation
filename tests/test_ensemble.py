"""
test_ensemble.py — 앙상블 랭커 테스트 (feature_engineering + ensemble_client)
=============================================================================
실행:
    cd D:/kride-project
    pytest tests/test_ensemble.py -v

외부 의존성 없음 (numpy만 필요). 모델 파일 없어도 fallback 테스트 가능.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml.feature_engineering import (
    haversine,
    compute_features,
    FEATURE_NAMES,
    PURPOSE_CATEGORY_MAP,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. haversine 거리 계산
# ═════════════════════════════════════════════════════════════════════════════
class TestHaversine:
    def test_same_point_zero_distance(self):
        assert haversine(37.5, 127.0, 37.5, 127.0) == 0.0

    def test_seoul_to_busan(self):
        """서울 ↔ 부산 약 325km"""
        dist = haversine(37.5665, 126.9780, 35.1796, 129.0756)
        assert 320 < dist < 340

    def test_symmetry(self):
        d1 = haversine(37.5, 127.0, 35.0, 129.0)
        d2 = haversine(35.0, 129.0, 37.5, 127.0)
        assert abs(d1 - d2) < 1e-6

    def test_known_distance_equator(self):
        """적도에서 경도 1도 차이 ≈ 111km"""
        dist = haversine(0.0, 0.0, 0.0, 1.0)
        assert 110 < dist < 112


# ═════════════════════════════════════════════════════════════════════════════
# 2. compute_features 단위 테스트
# ═════════════════════════════════════════════════════════════════════════════
class TestComputeFeatures:
    def _make_poi(self, **kwargs):
        base = {
            "poi_id": "p1",
            "name": "경복궁",
            "category": "관광지",
            "address": "서울 종로구",
            "lat": 37.58,
            "lon": 126.97,
        }
        base.update(kwargs)
        return base

    def test_output_shape(self):
        feats = compute_features(
            poi=self._make_poi(),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={},
        )
        assert feats.shape == (8,)
        assert feats.dtype == np.float32

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == 8

    def test_neo4j_hit_positive(self):
        feats = compute_features(
            poi=self._make_poi(),
            neo4j_poi_ids={"p1"},
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={},
        )
        assert feats[0] == 1.0  # neo4j_hit

    def test_neo4j_hit_by_name(self):
        """poi_id가 아닌 name으로도 매칭"""
        feats = compute_features(
            poi=self._make_poi(),
            neo4j_poi_ids={"경복궁"},
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={},
        )
        assert feats[0] == 1.0

    def test_neo4j_hit_negative(self):
        feats = compute_features(
            poi=self._make_poi(),
            neo4j_poi_ids={"other"},
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={},
        )
        assert feats[0] == 0.0

    def test_artist_count(self):
        feats = compute_features(
            poi=self._make_poi(),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={"p1": 3},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={},
        )
        assert feats[1] == 3.0  # neo4j_artist_count

    def test_chroma_similarity(self):
        feats = compute_features(
            poi=self._make_poi(),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={"p1": 0.85},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={},
        )
        assert feats[2] == pytest.approx(0.85)  # chroma_similarity

    def test_category_match_kculture(self):
        feats = compute_features(
            poi=self._make_poi(category="관광지"),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=["kculture"],
            user_budget={},
        )
        assert feats[4] == 1.0  # category_match

    def test_category_match_food(self):
        feats = compute_features(
            poi=self._make_poi(category="음식점"),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=["food"],
            user_budget={},
        )
        assert feats[4] == 1.0

    def test_category_no_match(self):
        feats = compute_features(
            poi=self._make_poi(category="음식점"),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=["nature"],
            user_budget={},
        )
        assert feats[4] == 0.0

    def test_region_match(self):
        feats = compute_features(
            poi=self._make_poi(address="서울 종로구"),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=["서울"],
            user_purposes=[],
            user_budget={},
        )
        assert feats[5] == 1.0  # region_match

    def test_region_no_match(self):
        feats = compute_features(
            poi=self._make_poi(address="부산 해운대구"),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=["서울"],
            user_purposes=[],
            user_budget={},
        )
        assert feats[5] == 0.0

    def test_distance_km(self):
        feats = compute_features(
            poi=self._make_poi(lat=37.58, lon=126.97),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={},
            user_lat=37.5,
            user_lon=127.0,
        )
        assert feats[6] > 0.0  # distance_km > 0

    def test_distance_no_coords(self):
        """좌표 미제공 시 distance=0"""
        feats = compute_features(
            poi=self._make_poi(),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={},
        )
        assert feats[6] == 0.0

    def test_budget_fit_in_range(self):
        feats = compute_features(
            poi=self._make_poi(avg_cost=50000),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={"min": 10000, "max": 100000},
        )
        assert feats[7] == 1.0  # budget_fit

    def test_budget_fit_out_of_range(self):
        feats = compute_features(
            poi=self._make_poi(avg_cost=200000),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={"min": 10000, "max": 100000},
        )
        assert feats[7] == 0.0

    def test_budget_fit_no_cost(self):
        """avg_cost 없으면 budget_fit=1 (통과)"""
        feats = compute_features(
            poi=self._make_poi(),
            neo4j_poi_ids=set(),
            neo4j_artist_counts={},
            chroma_similarities={},
            user_artists=[],
            user_regions=[],
            user_purposes=[],
            user_budget={"min": 10000, "max": 100000},
        )
        assert feats[7] == 1.0

    def test_all_features_together(self):
        """모든 feature가 동시에 올바르게 계산되는지"""
        feats = compute_features(
            poi=self._make_poi(category="관광지", avg_cost=50000),
            neo4j_poi_ids={"p1"},
            neo4j_artist_counts={"p1": 2},
            chroma_similarities={"p1": 0.9},
            user_artists=["BTS"],
            user_regions=["서울"],
            user_purposes=["kculture"],
            user_budget={"min": 0, "max": 100000},
            user_lat=37.5,
            user_lon=127.0,
        )
        assert feats[0] == 1.0   # neo4j_hit
        assert feats[1] == 2.0   # artist_count
        assert feats[2] == pytest.approx(0.9)  # chroma_sim
        assert feats[4] == 1.0   # category_match
        assert feats[5] == 1.0   # region_match
        assert feats[6] > 0.0    # distance
        assert feats[7] == 1.0   # budget_fit


# ═════════════════════════════════════════════════════════════════════════════
# 3. ensemble_client.rank_pois 테스트
# ═════════════════════════════════════════════════════════════════════════════
class TestEnsembleClient:
    @pytest.fixture(autouse=True)
    def reset_model(self):
        """각 테스트 전후 모델 캐시 초기화"""
        import src.api.ensemble_client as ec
        ec._model_data = None
        yield
        ec._model_data = None

    def test_empty_candidates(self):
        from src.api.ensemble_client import rank_pois
        result = rank_pois(
            neo4j_pois=[], chroma_pois=[],
            artists=[], regions=[], purposes=[], budget={},
        )
        assert result == []

    def test_fallback_no_model(self):
        """모델 파일 없으면 union fallback"""
        from src.api.ensemble_client import rank_pois
        pois = [
            {"poi_id": "p1", "name": "A", "category": "food", "address": "서울"},
            {"poi_id": "p2", "name": "B", "category": "nature", "address": "부산"},
        ]
        with patch("src.api.ensemble_client._MODEL_PATH", "/nonexistent/model.pkl"):
            result = rank_pois(
                neo4j_pois=pois, chroma_pois=[],
                artists=[], regions=[], purposes=[], budget={},
                top_k=5,
            )
        assert len(result) == 2
        # fallback 이므로 ensemble_score 없음
        assert "ensemble_score" not in result[0]

    def test_deduplication(self):
        """같은 poi_id 중복 제거"""
        from src.api.ensemble_client import rank_pois
        pois = [
            {"poi_id": "p1", "name": "A"},
            {"poi_id": "p1", "name": "A"},  # 중복
            {"poi_id": "p2", "name": "B"},
        ]
        with patch("src.api.ensemble_client._MODEL_PATH", "/nonexistent"):
            result = rank_pois(
                neo4j_pois=pois, chroma_pois=[],
                artists=[], regions=[], purposes=[], budget={},
            )
        assert len(result) == 2

    def test_with_mock_model(self):
        """mock 모델로 랭킹 검증"""
        import src.api.ensemble_client as ec

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5])
        ec._model_data = {"model": mock_model, "type": "lightgbm", "features": FEATURE_NAMES}

        neo4j_pois = [{"poi_id": "p1", "name": "High", "category": "food", "address": "서울"}]
        chroma_pois = [
            {"poi_id": "p2", "name": "Low", "category": "nature", "address": "부산", "similarity": 0.3},
            {"poi_id": "p3", "name": "Mid", "category": "food", "address": "서울", "similarity": 0.7},
        ]

        result = ec.rank_pois(
            neo4j_pois=neo4j_pois, chroma_pois=chroma_pois,
            artists=["BTS"], regions=["서울"],
            purposes=["food"], budget={"min": 0, "max": 500000},
            top_k=3,
        )

        assert len(result) == 3
        assert all("ensemble_score" in p for p in result)
        # 정렬 확인 (내림차순)
        scores = [p["ensemble_score"] for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self):
        """top_k 제한 검증"""
        import src.api.ensemble_client as ec

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ec._model_data = {"model": mock_model, "type": "lgbm", "features": FEATURE_NAMES}

        pois = [{"poi_id": f"p{i}", "name": f"POI_{i}", "category": "food", "address": "서울"} for i in range(5)]

        result = ec.rank_pois(
            neo4j_pois=pois, chroma_pois=[],
            artists=[], regions=[], purposes=[], budget={},
            top_k=2,
        )
        assert len(result) == 2

    def test_neo4j_chroma_merge(self):
        """Neo4j와 ChromaDB POI 합산"""
        from src.api.ensemble_client import rank_pois

        neo4j = [{"poi_id": "p1", "name": "A"}]
        chroma = [{"poi_id": "p2", "name": "B", "similarity": 0.8}]

        with patch("src.api.ensemble_client._MODEL_PATH", "/nonexistent"):
            result = rank_pois(
                neo4j_pois=neo4j, chroma_pois=chroma,
                artists=[], regions=[], purposes=[], budget={},
            )
        names = {p["name"] for p in result}
        assert names == {"A", "B"}


# ═════════════════════════════════════════════════════════════════════════════
# 4. build_ensemble_ranker 메트릭 함수 테스트
# ═════════════════════════════════════════════════════════════════════════════
class TestEnsembleMetrics:
    """ndcg_at_k, recall_at_k, map_at_k 함수 단위 테스트"""

    @pytest.fixture(autouse=True)
    def import_module(self):
        """build_ensemble_ranker.py에서 메트릭 함수만 가져오기"""
        try:
            # 필요한 stub
            for pkg in ["lightgbm", "xgboost", "mlflow", "dagshub"]:
                if pkg not in sys.modules:
                    sys.modules[pkg] = MagicMock()

            from src.ml.build_ensemble_ranker import ndcg_at_k, recall_at_k, map_at_k
            self.ndcg_at_k = ndcg_at_k
            self.recall_at_k = recall_at_k
            self.map_at_k = map_at_k
        except Exception:
            pytest.skip("build_ensemble_ranker import 실패")

    def test_ndcg_perfect_ranking(self):
        """완벽한 랭킹 → NDCG = 1.0"""
        # groups는 각 아이템의 그룹 ID (모두 그룹 0)
        y_true = np.array([3, 2, 1, 0, 0])
        y_pred = np.array([1.0, 0.8, 0.6, 0.2, 0.1])
        groups = np.array([0, 0, 0, 0, 0])
        score = self.ndcg_at_k(y_true, y_pred, groups, k=5)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_recall_at_k(self):
        """상위 K에 relevant 포함 비율"""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
        groups = np.array([0, 0, 0, 0, 0])
        score = self.recall_at_k(y_true, y_pred, groups, k=3)
        # 상위 3: idx [0,1,2] → relevant: idx 0,2 → recall = 2/3
        assert score == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_map_at_k(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.7, 0.5, 0.1])
        groups = np.array([0, 0, 0, 0])
        score = self.map_at_k(y_true, y_pred, groups, k=4)
        assert 0.0 <= score <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 5. PURPOSE_CATEGORY_MAP 검증
# ═════════════════════════════════════════════════════════════════════════════
class TestPurposeCategoryMap:
    def test_all_purposes_have_categories(self):
        expected_purposes = {"kculture", "food", "nature", "history", "shopping", "rest"}
        assert set(PURPOSE_CATEGORY_MAP.keys()) == expected_purposes

    def test_each_category_is_set(self):
        for purpose, cats in PURPOSE_CATEGORY_MAP.items():
            assert isinstance(cats, set), f"{purpose} 카테고리가 set이 아님"
            assert len(cats) > 0, f"{purpose} 카테고리가 비어있음"
