import numpy as np

from src.models.llm_explainer import (
    answer_user_chat,
    generate_executive_memorandum,
    generate_offline_memorandum,
)


def test_offline_memorandum_contains_all_selected_municipalities():
    selected_data = [
        {"name": "LOS SANTOS", "population": 12000, "prob_alto": 0.45, "cluster": "Nido Sísmico", "dist_nido": 8.5},
        {"name": "BUCARAMANGA", "population": 600000, "prob_alto": 0.20, "cluster": "Metrópoli", "dist_nido": 32.0},
        {"name": "ZAPATOCA", "population": 9000, "prob_alto": 0.35, "cluster": "Periferia Rural", "dist_nido": 18.2},
    ]
    cooldown = ["FLORIDABLANCA", "ARATOCA", "LEBRIJA"]

    doc = generate_offline_memorandum("2024-05", selected_data, cooldown)

    assert "LOS SANTOS" in doc
    assert "BUCARAMANGA" in doc
    assert "ZAPATOCA" in doc
    assert "FLORIDABLANCA" in doc  # Debe citar el cooldown
    assert "Ley 1523 de 2012" in doc
    assert len(doc) > 500


def test_executive_memorandum_with_numpy_array_cooldown():
    selected_data = [
        {"name": "LOS SANTOS", "population": 12000, "prob_alto": 0.45, "cluster": "Nido Sísmico", "dist_nido": 8.5},
        {"name": "BUCARAMANGA", "population": 600000, "prob_alto": 0.20, "cluster": "Metrópoli", "dist_nido": 32.0},
        {"name": "ZAPATOCA", "population": 9000, "prob_alto": 0.35, "cluster": "Periferia Rural", "dist_nido": 18.2},
    ]
    # Simula array numpy que disparaba el ValueError
    cooldown_np = np.array(["FLORIDABLANCA", "ARATOCA", "LEBRIJA"])

    doc, mode = generate_executive_memorandum(None, "2024-05", selected_data, cooldown_np)
    assert "LOS SANTOS" in doc
    assert "FLORIDABLANCA" in doc
    assert mode == "Motor de Inferencia Institucional (Modo Offline)"


def test_answer_user_chat_interactive():
    selected_data = [
        {"name": "LOS SANTOS", "population": 12000, "prob_alto": 0.45, "cluster": "Nido Sísmico", "dist_nido": 8.5},
        {"name": "BUCARAMANGA", "population": 600000, "prob_alto": 0.20, "cluster": "Metrópoli", "dist_nido": 32.0},
        {"name": "ZAPATOCA", "population": 9000, "prob_alto": 0.35, "cluster": "Periferia Rural", "dist_nido": 18.2},
    ]
    reply_why, _ = answer_user_chat(None, "¿Por qué se priorizaron estos municipios?", "2024-05", selected_data, ["FLORIDABLANCA"])
    assert "LOS SANTOS" in reply_why
    assert "PPO" in reply_why

    reply_cool, _ = answer_user_chat(None, "¿Por qué hay municipios en cooldown?", "2024-05", selected_data, ["FLORIDABLANCA"])
    assert "cooldown" in reply_cool.lower() or "descanso" in reply_cool.lower()


def test_executive_memorandum_fallback_without_api_key():
    selected_data = [
        {"name": "ARATOCA", "population": 8500, "prob_alto": 0.30, "cluster": "Nido Sísmico", "dist_nido": 12.0},
        {"name": "GIRÓN", "population": 180000, "prob_alto": 0.15, "cluster": "Polos Urbanos", "dist_nido": 35.0},
        {"name": "BETULIA", "population": 5000, "prob_alto": 0.28, "cluster": "Nido Sísmico", "dist_nido": 15.0},
    ]
    doc, mode = generate_executive_memorandum("", "2023-11", selected_data, ["LOS SANTOS"])
    assert "ARATOCA" in doc
    assert "GIRÓN" in doc
    assert mode == "Motor de Inferencia Institucional (Modo Offline)"


def test_answer_user_chat_los_santos_nest_query():
    selected_data = [
        {"name": "LOS SANTOS", "population": 12000, "prob_alto": 0.45, "cluster": "Nido Sísmico", "dist_nido": 8.5},
        {"name": "BUCARAMANGA", "population": 600000, "prob_alto": 0.20, "cluster": "Metrópoli", "dist_nido": 32.0},
        {"name": "ZAPATOCA", "population": 9000, "prob_alto": 0.35, "cluster": "Periferia Rural", "dist_nido": 18.2},
    ]
    reply, src = answer_user_chat(None, "Por que los santos es el lugar que mas tiembla", "2024-05", selected_data, ["FLORIDABLANCA"])
    assert "Nido Sísmico" in reply
    assert "Los Santos" in reply
    assert "Modo Offline" in src


def test_validate_gemini_key_empty():
    from src.models.llm_explainer import validate_gemini_key
    is_valid, msg = validate_gemini_key("")
    assert not is_valid
    assert "Sin API Key" in msg
