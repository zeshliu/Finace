from __future__ import annotations

from src.industry import enrich_candidate_industries
from src.providers import normalize_industry_name


def test_normalize_industry_name():
    assert normalize_industry_name("C39计算机、通信和其他电子设备制造业") == "电子设备"
    assert normalize_industry_name("D44电力、热力生产和供应业") == "电力"
    assert normalize_industry_name("半导体") == "半导体"
    assert normalize_industry_name("") == "未分类"


def test_enrich_candidate_industries_with_fallback(tmp_path):
    class Provider:
        def get_industry_map_baostock(self):
            return {"688981": "电子设备", "600900": "电力"}

        def get_industry(self, code):
            if code == "688981":
                return "半导体"
            raise RuntimeError("细分行业暂不可用")

    candidates = [{"code": "688981"}, {"code": "600900"}]
    result = enrich_candidate_industries(candidates, Provider(), tmp_path / "industry.json", 2)
    assert result[0]["industry"] == "半导体"
    assert result[1]["industry"] == "电力"
