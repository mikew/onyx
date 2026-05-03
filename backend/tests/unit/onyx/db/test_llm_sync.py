"""Tests for LLM provider model sync functionality."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from onyx.db.llm import sync_model_configurations
from onyx.llm.constants import LlmProviderNames
from onyx.server.manage.llm.models import SyncModelEntry


def _make_provider(
    *existing_models: MagicMock,
    provider_id: int = 1,
) -> MagicMock:
    mock_provider = MagicMock()
    mock_provider.id = provider_id
    mock_provider.model_configurations = list(existing_models)
    return mock_provider


def _make_existing_model(
    name: str,
    flow_types: list[LLMModelFlowType],
    supports_image_input: bool = False,
) -> MagicMock:
    """Create a mock ModelConfiguration with real Python collections so that
    ``in`` / ``not in`` checks and set operations work correctly."""
    mc = MagicMock()
    mc.name = name
    mc.llm_model_flow_types = list(flow_types)
    mc.supports_image_input = supports_image_input
    return mc


class TestSyncModelConfigurations:
    """Tests for sync_model_configurations."""

    # ------------------------------------------------------------------
    # New-model insertion (unchanged behaviour)
    # ------------------------------------------------------------------

    def test_inserts_new_models(self) -> None:
        mock_session = MagicMock()
        provider = _make_provider()

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        max_input_tokens=128000,
                        supports_image_input=True,
                    ),
                    SyncModelEntry(
                        name="gpt-4o",
                        display_name="GPT-4o",
                        max_input_tokens=128000,
                        supports_image_input=True,
                    ),
                ],
            )

        assert result == 2
        # 2 models × (1 ModelConfiguration INSERT + 1 CHAT flow + 1 VISION flow)
        assert mock_session.execute.call_count == 2 * 3
        mock_session.commit.assert_called_once()

    def test_raises_on_missing_provider(self) -> None:
        mock_session = MagicMock()

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                sync_model_configurations(
                    db_session=mock_session,
                    provider_name="nonexistent",
                    models=[SyncModelEntry(name="model", display_name="Model")],
                )

    def test_inserts_reasoning_flow_for_new_model(self) -> None:
        mock_session = MagicMock()
        provider = _make_provider()

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="deepseek-r1",
                        display_name="DeepSeek R1",
                        max_input_tokens=65536,
                        supports_image_input=True,
                        supports_reasoning=True,
                    ),
                ],
            )

        assert result == 1
        # 1 ModelConfiguration INSERT + 3 flow inserts (CHAT + VISION + REASONING)
        assert mock_session.execute.call_count == 4
        mock_session.commit.assert_called_once()

    # ------------------------------------------------------------------
    # Flow reconciliation for existing models
    # ------------------------------------------------------------------

    def test_adds_missing_flow_to_existing_model(self) -> None:
        """Existing model has CHAT only; API reports VISION → VISION flow added."""
        existing = _make_existing_model(
            "gpt-4", flow_types=[LLMModelFlowType.CHAT], supports_image_input=False
        )
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_image_input=True,
                    ),
                ],
            )

        assert result == 0  # no new models
        mock_session.execute.assert_called_once()  # one INSERT for the new VISION flow
        mock_session.commit.assert_called_once()

    def test_adds_missing_reasoning_flow_to_existing_model(self) -> None:
        """Existing model has CHAT only; API now reports supports_reasoning → REASONING added."""
        existing = _make_existing_model("o1", flow_types=[LLMModelFlowType.CHAT])
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="o1",
                        display_name="O1",
                        supports_reasoning=True,
                    ),
                ],
            )

        assert result == 0
        mock_session.execute.assert_called_once()  # INSERT for REASONING flow
        mock_session.commit.assert_called_once()

    def test_removes_flow_no_longer_supported(self) -> None:
        """Existing model has CHAT+VISION; API no longer reports VISION → VISION row deleted."""
        existing = _make_existing_model(
            "gpt-4",
            flow_types=[LLMModelFlowType.CHAT, LLMModelFlowType.VISION],
            supports_image_input=True,
        )
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_image_input=False,
                    ),
                ],
            )

        assert result == 0
        mock_session.execute.assert_called_once()  # DELETE for the stale VISION flow
        mock_session.commit.assert_called_once()

    def test_mirrors_flow_set_exactly(self) -> None:
        """sync([CHAT, VISION]) → sync([CHAT, REASONING]) should leave exactly CHAT+REASONING.
        REASONING is added, VISION is deleted, CHAT stays untouched."""
        existing = _make_existing_model(
            "gpt-4",
            flow_types=[LLMModelFlowType.CHAT, LLMModelFlowType.VISION],
            supports_image_input=True,
        )
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_reasoning=True,
                        supports_image_input=False,
                    ),
                ],
            )

        # 1 INSERT (add REASONING) + 1 DELETE (remove VISION); CHAT unchanged
        assert mock_session.execute.call_count == 2
        mock_session.commit.assert_called_once()

    def test_does_not_modify_flows_already_in_sync(self) -> None:
        """Existing model already matches the API — no execute calls for flows."""
        existing = _make_existing_model(
            "gpt-4",
            flow_types=[LLMModelFlowType.CHAT, LLMModelFlowType.VISION],
            supports_image_input=True,
        )
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_image_input=True,
                    ),
                ],
            )

        assert result == 0
        mock_session.execute.assert_not_called()
        mock_session.commit.assert_not_called()

    # ------------------------------------------------------------------
    # Conditional commit
    # ------------------------------------------------------------------

    def test_commit_called_when_flows_added(self) -> None:
        """No new models, but a missing flow is added → commit is called."""
        existing = _make_existing_model("gpt-4", flow_types=[LLMModelFlowType.CHAT])
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_image_input=True,
                    ),
                ],
            )

        mock_session.commit.assert_called_once()

    def test_commit_called_when_flows_removed(self) -> None:
        """No new models, but a stale flow is removed → commit is called."""
        existing = _make_existing_model(
            "gpt-4",
            flow_types=[LLMModelFlowType.CHAT, LLMModelFlowType.VISION],
            supports_image_input=True,
        )
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_image_input=False,
                    ),
                ],
            )

        mock_session.commit.assert_called_once()

    def test_commit_not_called_when_nothing_changed(self) -> None:
        """Existing model already matches the API — commit must NOT be called."""
        existing = _make_existing_model(
            "gpt-4",
            flow_types=[LLMModelFlowType.CHAT],
            supports_image_input=False,
        )
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_image_input=False,
                    ),
                ],
            )

        assert result == 0
        mock_session.commit.assert_not_called()

    # ------------------------------------------------------------------
    # supports_image_input denormalized column
    # ------------------------------------------------------------------

    def test_sets_supports_image_input_true(self) -> None:
        """existing.supports_image_input=False, API says True → flipped to True."""
        existing = _make_existing_model(
            "gpt-4", flow_types=[LLMModelFlowType.CHAT], supports_image_input=False
        )
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_image_input=True,
                    ),
                ],
            )

        assert existing.supports_image_input is True

    def test_clears_supports_image_input_false(self) -> None:
        """existing.supports_image_input=True, API no longer reports it → flipped to False."""
        existing = _make_existing_model(
            "gpt-4",
            flow_types=[LLMModelFlowType.CHAT, LLMModelFlowType.VISION],
            supports_image_input=True,
        )
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",
                        display_name="GPT-4",
                        supports_image_input=False,
                    ),
                ],
            )

        assert existing.supports_image_input is False

    # ------------------------------------------------------------------
    # Mixed new + existing
    # ------------------------------------------------------------------

    def test_mixed_new_and_existing_with_flow_reconciliation(self) -> None:
        """One new model + one existing model needing a new flow — both handled correctly."""
        existing = _make_existing_model("gpt-4", flow_types=[LLMModelFlowType.CHAT])
        mock_session = MagicMock()
        provider = _make_provider(existing)

        with patch("onyx.db.llm.fetch_existing_llm_provider", return_value=provider):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_name=LlmProviderNames.OPENAI,
                models=[
                    SyncModelEntry(
                        name="gpt-4",  # existing — needs VISION added
                        display_name="GPT-4",
                        supports_image_input=True,
                    ),
                    SyncModelEntry(
                        name="gpt-4o",  # new
                        display_name="GPT-4o",
                        supports_image_input=True,
                    ),
                ],
            )

        assert result == 1  # one new model
        # 1 VISION flow INSERT (reconcile) + 3 executes for gpt-4o (model + CHAT + VISION)
        assert mock_session.execute.call_count == 4
        mock_session.commit.assert_called_once()
