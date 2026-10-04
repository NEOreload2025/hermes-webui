"""Regression tests for the 2026-10-05 Nous free-tier picker fix.

Two real-world breakages NEO hit after connecting a Nous Portal *Free*
account (hinet.jc@gmail.com) on 2026-10-05:

1. The Nous picker group advertised 425 catalog entries of which 425-minus-5
   would 404 ``insufficient_credits_for_paid_model`` at inference time — the
   dropdown (15 featured) AND the extras bucket (410) were all paid models on
   a Free account. The picker is gated on ``logged_in`` only, so paid vs free
   was invisible to it.

2. ``model_catalog.excluded_providers`` (a countersigned config policy gate,
   NEO decision 2026-10-04) listed ``opencode-free``, yet the WebUI picker
   still rendered a 12-model OpenCode Free group — the WebUI never respected
   the agent's catalog exclusion list.

Fix shape:
- ``_build_nous_featured_set(free_gate=True)`` excludes paid ids when the
  account can't call them (``:free`` ids fill the catalog first; the paid
  vendor-priority top-up only runs if free ids can't reach ``target``).
- ``_config_excluded_providers()`` reads the agent's exclusion list and both
  catalog build paths drop those groups after sorting.
"""

from __future__ import annotations

import api.config as config


# ────────────────────────────────────────────────────────────────────────
# _config_excluded_providers
# ────────────────────────────────────────────────────────────────────────
class TestConfigExcludedProviders:
    def test_reads_agent_exclusion_list(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "cfg",
            {"model_catalog": {"excluded_providers": ["copilot", "opencode-free"]}},
        )
        assert config._config_excluded_providers() == {"copilot", "opencode-free"}

    def test_missing_config_means_no_filtering(self, monkeypatch):
        monkeypatch.setattr(config, "cfg", {})
        assert config._config_excluded_providers() == set()

    def test_non_list_value_is_ignored(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "cfg",
            {"model_catalog": {"excluded_providers": "copilot"}},
        )
        assert config._config_excluded_providers() == set()

    def test_slugs_are_canonicalised(self, monkeypatch):
        # alias-table spellings of the same provider collapse to one slug
        monkeypatch.setattr(
            config,
            "cfg",
            {"model_catalog": {"excluded_providers": ["opencode-free", "opencode_free"]}},
        )
        assert config._config_excluded_providers() == {"opencode-free"}


# ────────────────────────────────────────────────────────────────────────
# Nous free-tier gating helpers
# ────────────────────────────────────────────────────────────────────────
class TestNousFreeTierHelpers:
    def test_free_tier_detected_when_no_billing_scope(self, monkeypatch):
        import sys
        import types

        fake_auth = types.ModuleType("hermes_cli.auth")
        setattr(fake_auth, "nous_token_has_billing_scope", lambda: False)
        monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth)
        assert config._nous_free_tier_account() is True

    def test_paid_tier_when_billing_scope_present(self, monkeypatch):
        import sys
        import types

        fake_auth = types.ModuleType("hermes_cli.auth")
        setattr(fake_auth, "nous_token_has_billing_scope", lambda: True)
        monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth)
        assert config._nous_free_tier_account() is False

    def test_hermes_cli_unavailable_means_paid_passthrough(self, monkeypatch):
        # hermes_cli not importable → must NOT gate (worst case: show the
        # full catalog, never hide a paid account's usable models)
        import builtins

        real_import = builtins.__import__

        def _fail(name, *a, **k):
            if name.startswith("hermes_cli"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _fail)
        assert config._nous_free_tier_account() is False

    def test_free_usability_detected_from_live_ids(self):
        live = ["anthropic/claude-opus-5.5", "meituan/longcat-2.5-preview:free"]
        assert config._nous_models_usable_on_free_tier(live) is True

    def test_paid_only_catalog_has_no_free_usability(self):
        live = ["anthropic/claude-opus-5.5", "openai/gpt-6-luna"]
        assert config._nous_models_usable_on_free_tier(live) is False


# ────────────────────────────────────────────────────────────────────────
# _build_nous_featured_set with free_gate
# ────────────────────────────────────────────────────────────────────────
_FREE_CATALOG = [
    "inclusionai/ling-3.0-flash-sante:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "meituan/longcat-2.5-preview:free",
    "stepfun/step-3.7-flash:free",
    # …plus the 400+ paid entries a Free account sees but cannot call
    *[f"anthropic/claude-paid-{i}" for i in range(10)],
    *[f"openai/gpt-paid-{i}" for i in range(10)],
]


class TestNousFeaturedSetFreeGate:
    def test_free_gate_excludes_paid_from_both_buckets(self):
        featured, extras = config._build_nous_featured_set(
            _FREE_CATALOG, free_gate=True
        )
        everything = set(featured) | set(extras)
        assert not any("paid" in mid for mid in everything)
        assert set(_FREE_CATALOG[:5]) == everything  # exactly the free ids

    def test_free_gate_drops_paid_flagships(self):
        # The static flagships are paid (claude-opus-4.6 etc.); with the gate
        # on they must not survive as featured entries.
        catalog = [
            "anthropic/claude-opus-4.6",
            "meituan/longcat-2.5-preview:free",
        ]
        featured, _ = config._build_nous_featured_set(catalog, free_gate=True)
        assert featured == ["meituan/longcat-2.5-preview:free"]

    def test_no_gate_keeps_full_catalog(self):
        featured, extras = config._build_nous_featured_set(
            ["anthropic/claude-opus-4.6", "stepfun/step-3.7-flash:free"],
            free_gate=False,
        )
        assert featured == ["anthropic/claude-opus-4.6", "stepfun/step-3.7-flash:free"]
        assert extras == []

    def test_paid_account_catalog_never_gated_even_with_gate_flag(self):
        # No :free ids in the catalog → paid plan → pass through unfiltered
        catalog = [f"anthropic/claude-{i}" for i in range(30)]
        featured, extras = config._build_nous_featured_set(catalog, free_gate=True)
        assert len(featured) == config._NOUS_FEATURED_TARGET
        assert len(featured) + len(extras) == 30

    def test_free_catalog_below_threshold_stays_below_target(self):
        # 3 free ids + 30 paid: gate keeps only the 3 free ids (featured),
        # paid ids vanish from BOTH buckets. Dropdown shows 3 usable entries
        # instead of 3 usable + 12 dead ones.
        catalog = [
            "meituan/longcat-2.5-preview:free",
            "stepfun/step-3.7-flash:free",
            "poolside/laguna-s-2.1:free",
            *[f"openai/gpt-{i}" for i in range(30)],
        ]
        featured, extras = config._build_nous_featured_set(
            catalog, free_gate=True
        )
        assert featured == catalog[:3]
        assert extras == []
