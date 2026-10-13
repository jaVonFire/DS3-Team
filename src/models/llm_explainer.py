"""Módulo Explicador con Modelos de Lenguaje (LLM) — Nivel Avanzado (Impacto e Innovación).

Traduce las salidas cuantitativas del pipeline encadenado (predicciones XGBoost, valores SHAP,
perfil de clustering territorial y decisiones del agente PPO) en un Memorando Ejecutivo
de Alerta Temprana y Asignación de Recursos para la Gobernación de Santander y alcaldes locales.

Soporta:
1. Conexión en vivo con Google Gemini (requiere API Key ingresada por el usuario o en entorno).
2. Motor de Inferencia Institucional Local (Offline fallback de alta calidad) para garantizar
   que la sustentación nunca falle si no hay conexión a internet o API Key.
"""

from __future__ import annotations

import json
import os
from typing import Any
import requests


import re

def _extract_recommended_model(error_text: str) -> str | None:
    """Extrae el modelo sugerido por Google si el error contiene 'use models/...'."""
    match = re.search(r"use models/([a-zA-Z0-9\.\-_]+)", error_text)
    if match:
        return match.group(1)
    return None


def discover_active_gemini_model(key: str) -> tuple[str, str]:
    """Consulta ModelService.ListModels para encontrar el modelo exacto disponible para la clave del usuario."""
    for api_ver in ["v1beta", "v1"]:
        try:
            list_url = f"https://generativelanguage.googleapis.com/{api_ver}/models?key={key}"
            res = requests.get(list_url, timeout=7)
            if res.status_code == 200:
                models_data = res.json().get("models", [])
                valid_models = [
                    m["name"] for m in models_data 
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                # Priorizar modelos más recientes de 2026: 3.8-flash, 3.0, etc.
                for pattern in ["3.8-flash", "gemini-3", "2.5-flash", "2.0-flash", "1.5-flash", "flash", "gemini"]:
                    for m_name in valid_models:
                        if pattern in m_name.lower():
                            return api_ver, m_name.replace("models/", "")
                if valid_models:
                    return api_ver, valid_models[0].replace("models/", "")
        except Exception:
            continue
    return "v1beta", "gemini-3.8-flash"


def call_gemini_api(api_key: str, prompt: str, timeout: int = 15) -> tuple[str | None, str]:
    """Invoca la API de Google Gemini mediante el modelo descubierto dinámicamente con auto-recuperación."""
    key = (api_key or "").strip().strip('"').strip("'")
    if not key:
        return None, "No se proporcionó API Key"

    api_ver, primary_model = discover_active_gemini_model(key)
    candidate_models = [primary_model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-pro"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 900,
        },
    }

    last_err = ""
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent?key={key}"
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip(), f"Google {model_name} (Online)"
            
            err_json = response.json().get("error", {})
            err_msg = err_json.get("message", f"HTTP {response.status_code}")
            last_err = err_msg

            # Si el error indica otro modelo recomendado, probarlo
            rec_model = _extract_recommended_model(err_msg)
            if rec_model and rec_model not in models_to_try:
                models_to_try.append(rec_model)

            # Si es por alta demanda, continuar con el siguiente modelo candidato
            if "high demand" in err_msg.lower() or "overloaded" in err_msg.lower() or response.status_code in (429, 503):
                continue
        except Exception as e:
            last_err = str(e)
            continue

    return None, f"Google Cloud temporalmente saturado: {last_err}"


def validate_gemini_key(api_key: str | None) -> tuple[bool, str]:
    """Valida de forma rápida si una API Key de Gemini es funcional descubriendo el modelo activo."""
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip().strip('"').strip("'")
    if not key:
        return False, "Sin API Key ingresada (Modo Offline activo)"

    api_ver, primary_model = discover_active_gemini_model(key)
    candidate_models = [primary_model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    payload = {
        "contents": [{"parts": [{"text": "OK"}]}],
        "generationConfig": {"maxOutputTokens": 2},
    }

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent?key={key}"
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=8)
            if res.status_code == 200:
                return True, f"Conexión exitosa con Google {model_name} (Online)"
            
            err = res.json().get("error", {}).get("message", f"HTTP {res.status_code}")
            # Si el error es por alta demanda, la clave ES VÁLIDA pero los servidores de Google están bajo congestión
            if "high demand" in err.lower() or "overloaded" in err.lower() or res.status_code in (429, 503):
                return True, f"Clave válida · Google {model_name} con alta demanda temporal (Motor Offline activo como respaldo)"
            
            rec = _extract_recommended_model(err)
            if rec and rec not in models_to_try:
                models_to_try.append(rec)
        except Exception as e:
            continue

    return False, "No se pudo conectar con los servidores de Google Gemini"


def _normalize_cooldown_names(cooldown_names: Any) -> list[str]:
    """Convierte cualquier iterable, array numpy o Series a una lista limpia de strings."""
    if cooldown_names is None:
        return []
    try:
        if hasattr(cooldown_names, "tolist"):
            items = cooldown_names.tolist()
        else:
            items = list(cooldown_names)
        return [str(x) for x in items if str(x).strip()]
    except Exception:
        return [str(cooldown_names)]


def generate_offline_memorandum(
    period: str,
    selected_data: list[dict[str, Any]],
    cooldown_names: Any,
) -> str:
    """Genera un memorando ejecutivo estructurado con rigurosidad técnica bajo la Ley 1523 de 2012."""
    nombres_priorizados = [d["name"] for d in selected_data]
    total_pob = sum(d.get("population", 0) for d in selected_data)
    c_list = _normalize_cooldown_names(cooldown_names)
    cooldown_str = ", ".join(c_list) if len(c_list) > 0 else "Ninguno (inicio del ciclo)"

    items_justificacion = []
    for idx, d in enumerate(selected_data, 1):
        name = d["name"]
        pob = d.get("population", 0)
        p_alto = d.get("prob_alto", 0.0) * 100
        cluster = d.get("cluster", "Territorio Departamental")
        dist = d.get("dist_nido", 50.0)

        item = (
            f"**{idx}. Municipio de {name}** (Población expuesta: {pob:,.0f} hab | Perfil: *{cluster}*):\n"
            f"   - **Factor Amenaza:** Probabilidad estimada de sismicidad crítica del {p_alto:.1f}% en el mes subsiguiente, con una distancia epicentral de {dist:.1f} km al Nido Sísmico de Bucaramanga.\n"
            f"   - **Factor Exposición:** Concentración de equipamientos educativos y hospitalarios en cabecera urbana con requerimiento prioritario de verificación sismorresistente.\n"
            f"   - **Acción Inmediata:** Activación de comités locales (CMGRD), inventario de albergues y simulación de protocolos de telecomunicaciones."
        )
        items_justificacion.append(item)

    justificaciones_str = "\n\n".join(items_justificacion)

    memorando = f"""# 🏛️ REPUBLICA DE COLOMBIA — GOBERNACIÓN DE SANTANDER
### CONSEJO DEPARTAMENTAL PARA LA GESTIÓN DEL RIESGO DE DESASTRES (CDGRD)
**MEMORANDO EXTRAORDINARIO DE ASIGNACIÓN TERRITORIAL SÍSMICA · PERIODO: {period}**  
*Marco Normativo: Ley 1523 de 2012 (Política Nacional de Gestión del Riesgo de Desastres)*

---

### 1. Resumen Ejecutivo de la Decisión
En cumplimiento del principio de **Priorización Basada en Evidencia y Equidad Territorial**, el Sistema de Decisión Secuencial del Departamento ha determinado la asignación prioritaria de los 3 cupos mensuales de recursos preventivos a los municipios de **{', '.join(nombres_priorizados)}**, salvaguardando un total acumulado de **{total_pob:,.0f} vidas humanas**.

### 2. Fundamentación Técnica y Desglose por Municipio
{justificaciones_str}

### 3. Principio de Rotación y Restricción de Cooldown
- **Municipios en Periodo de Descanso Operativo (Cooldown):** **{cooldown_str}**.
- **Justificación:** Habiendo recibido asignaciones en el ciclo mensual inmediatamente anterior, la política estricta del Agente de Refuerzo prohíbe la duplicidad consecutiva de fondos. Esto garantiza que el presupuesto no colapse en un solo territorio y protege a los municipios periféricos contra el abandono acumulado (>6 meses).

### 4. Instrucción Operativa para los Consejos Municipales (CMGRD)
1. Instalar salas de crisis preventivas en las alcaldías priorizadas antes del inicio del mes.
2. Coordinar con la Unidad Departamental de Gestión del Riesgo el despliegue de maquinaria amarilla en puntos críticos de conectividad vial.
3. Emitir boletines comunitarios pedagógicos enfatizando que esta medida es **preventiva y de preparación**, no una predicción determinista de catástrofe.

---
*Emitido por el Sistema Integrado de Inteligencia Sísmica Territorial (DS3 Team · CIAI-UTS).*
"""
    return memorando.strip()


def build_llm_prompt(
    period: str,
    selected_data: list[dict[str, Any]],
    cooldown_names: Any,
) -> str:
    """Construye el prompt estructurado para invocar a Gemini con el contexto de Santander."""
    mun_summary = []
    for d in selected_data:
        mun_summary.append(
            f"- Municipio: {d['name']}, Población: {d.get('population', 0):,.0f}, Probabilidad Riesgo Alto: {d.get('prob_alto', 0.0):.2%}, "
            f"Perfil Clúster: {d.get('cluster', 'N/A')}, Distancia Nido Sísmico: {d.get('dist_nido', 0.0):.1f} km"
        )
    muns_str = "\n".join(mun_summary)
    c_list = _normalize_cooldown_names(cooldown_names)
    cooldown_str = ", ".join(c_list) if len(c_list) > 0 else "Ninguno"

    prompt = f"""Actúa como el Director del Consejo Departamental para la Gestión del Riesgo de Desastres de Santander (Colombia).
Redacta un Memorando Ejecutivo Oficial de Alerta Temprana y Priorización de Recursos Sísmicos para el mes {period}.

El sistema inteligente (compuesto por un clasificador XGBoost, clustering territorial y un agente de refuerzo PPO) seleccionó los siguientes 3 municipios:
{muns_str}

Restricciones operativas del sistema que debes explicar:
- Municipios en descanso (cooldown) por ser atendidos el mes anterior: {cooldown_str}.
- Regla: La recompensa pondera Riesgo = Amenaza Sísmica x Exposición de Población (DANE).
- Regla: Penalización severa si un municipio pasa más de 6 meses abandonado.

Estructura obligatoria del memorando:
1. Encabezado institucional formal de la Gobernación de Santander citando la Ley 1523 de 2012.
2. Resumen ejecutivo de por qué se priorizan estos 3 municipios en este mes específico.
3. Justificación técnica de cada municipio (mencionando sismicidad, vidas expuestas e infraestructura crítica como colegios y hospitales).
4. Explicación de por qué los municipios en cooldown ({cooldown_str}) no reciben recursos este mes para evitar favoritismos y colapso de política.
5. Recomendaciones prácticas e inmediatas para los alcaldes y Comités Municipales (CMGRD).

Usa un tono altamente profesional, formal, preventivo y claro para autoridades municipales. No uses lenguaje robótico."""
    return prompt


def generate_executive_memorandum(
    api_key: str | None,
    period: str,
    selected_data: list[dict[str, Any]],
    cooldown_names: Any,
) -> tuple[str, str]:
    """Genera el memorando ejecutivo intentando primero Gemini online y recurriendo al motor institucional."""
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip().strip('"').strip("'")

    if key:
        prompt = build_llm_prompt(period, selected_data, cooldown_names)
        gemini_response, model_source = call_gemini_api(key, prompt)
        if gemini_response:
            return gemini_response, model_source

    # Fallback transparente y robusto
    offline_doc = generate_offline_memorandum(period, selected_data, cooldown_names)
    fallback_note = "Motor de Inferencia Institucional (Modo Offline)"
    return offline_doc, fallback_note


def answer_user_chat(
    api_key: str | None,
    user_query: str,
    period: str,
    selected_data: list[dict[str, Any]],
    cooldown_names: Any,
) -> tuple[str, str]:
    """Responde consultas en lenguaje natural como un agente experto del CDGRD Santander."""
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip().strip('"').strip("'")
    c_list = _normalize_cooldown_names(cooldown_names)
    cooldown_str = ", ".join(c_list) if len(c_list) > 0 else "Ninguno"
    muns_names = [d["name"] for d in selected_data]
    total_pob = sum(d.get("population", 0) for d in selected_data)

    if key:
        context_items = [
            f"- {d['name']} (Pob: {d.get('population', 0):,.0f}, Prob. Alto: {d.get('prob_alto', 0.0):.1%}, Clúster: {d.get('cluster', 'N/A')}, Dist. Nido: {d.get('dist_nido', 0):.1f}km)"
            for d in selected_data
        ]
        chat_prompt = f"""Eres el Asistente Experto en Gestión del Riesgo Sísmico de la Gobernación de Santander (Consejo Departamental de Gestión del Riesgo - CDGRD, Ley 1523 de 2012).
Trabajas con un sistema de toma de decisiones secuencial desarrollado por el DS3 Team para la Hackatón Sismos Santander (Agente PPO con datos SGC, DANE e IGAC).

Contexto del mes {period}:
- Municipios priorizados con presupuesto: {', '.join(muns_names)} (Total población protegida: {total_pob:,.0f} hab)
- Detalle técnico:
{chr(10).join(context_items)}
- Municipios en descanso operativo (cooldown del mes anterior): {cooldown_str}
- Política de decisión: Maximiza Riesgo = Amenaza Sísmica x Exposición Humana con penalización estricta por abandono territorial (>6 meses).
- Nido Sísmico de Bucaramanga: epicentro histórico en Los Santos (140-160 km de profundidad), concentración de más del 60% de sismos del país.

Pregunta de la autoridad territorial o evaluador:
"{user_query}"

Instrucciones:
Responde de manera muy clara, profesional, institucional y empática en 1 o 2 párrafos concisos. Explica las razones técnicas y humanas de la decisión sin tecnicismos innecesarios."""
        gemini_reply, model_source = call_gemini_api(key, chat_prompt)
        if gemini_reply:
            return gemini_reply, model_source
        else:
            # Fallback elegante y limpio
            offline_source = "Motor Experto Institucional (Modo Offline · Respaldo por Saturación en la Nube)"
    else:
        offline_source = "Motor Experto Institucional (Modo Offline - Sin API Key)"

    # Motor conversacional offline contextual de alta precisión
    q_lower = user_query.lower()

    if any(k in q_lower for k in ["santos", "nido", "tiembla", "mas tiembla", "más tiembla", "frecuencia", "epicentro"]):
        reply = (
            "**Los Santos** es el epicentro geográfico del **Nido Sísmico de Bucaramanga**, la segunda región con mayor actividad sísmica del planeta "
            "(después del Nido de Vrancea en Rumania). Concentra más del **60% de los sismos registrados en toda Colombia**, promediando entre 20 y 40 eventos diarios. "
            "Esto se origina en un atrapamiento tectónico donde convergen las placas de Nazca y el Caribe bajo la placa Sudamericana a una profundidad intermedia de **140 a 160 kilómetros**.\n\n"
            "Sin embargo, debido a esa gran profundidad, la energía se disipa antes de alcanzar la corteza superficial. Por ello, nuestro modelo aplica la doctrina de la Ley 1523: "
            "**Riesgo = Amenaza × Exposición**. Aunque en Los Santos tiembla más veces, un sismo allí genera menor riesgo humanitario comparado con sismos superficiales o sacudidas "
            "en urbes densamente pobladas como Bucaramanga, Girón o Piedecuesta, donde se concentran colegios, hospitales y construcciones vulnerables."
        )
    elif any(k in q_lower for k in ["por qué", "porque", "por que", "razon", "motivo", "prioriz", "elig"]):
        reply = (
            f"Para el periodo **{period}**, el Agente PPO seleccionó a **{', '.join(muns_names)}** "
            f"porque combinan una alta amenaza sísmica proyectada por el clasificador XGBoost con una concentración crítica "
            f"de **{total_pob:,.0f} habitantes (DANE)** y equipamientos esenciales (hospitales y colegios). "
            f"Además, se respetó la regla de descanso sobre **{cooldown_str}** para evitar la concentración repetitiva del presupuesto."
        )
    elif any(k in q_lower for k in ["cooldown", "descanso", "repetir", "anterior", "por que no"]):
        reply = (
            f"La regla de **cooldown operativo** prohíbe que un municipio reciba asignaciones presupuestales en dos meses consecutivos. "
            f"En este mes, **{cooldown_str}** está en periodo de descanso porque fue atendido el mes inmediatamente anterior. "
            f"Esto es fundamental: si no tuviéramos esta restricción, Bucaramanga acapararía el 100% de los recursos todos los meses por su tamaño, "
            f"dejando desprotegida a la provincia santandereana."
        )
    elif any(k in q_lower for k in ["poblacion", "dane", "exposicion", "amenaza", "formula"]):
        reply = (
            f"Nuestro modelo aplica el principio rector de la Ley 1523 de 2012: **Riesgo = Amenaza × Exposición**. "
            f"No basta con saber dónde tiembla (sismicidad física); lo que realmente genera desastre es el impacto en vidas e infraestructura. "
            f"Por ello, la función de recompensa del agente multiplica la energía sísmica por el factor logarítmico poblacional del Censo DANE, "
            f"protegiendo en **{period}** a más de **{total_pob:,.0f} ciudadanos**."
        )
    elif any(k in q_lower for k in ["ley", "norma", "juridico", "1523", "legal"]):
        reply = (
            f"Bajo el artículo 3 de la **Ley 1523 de 2012**, la gestión del riesgo es un proceso social orientado a la formulación y ejecución de políticas "
            f"de conocimiento, reducción del riesgo y manejo de desastres. Nuestro sistema dota a los Consejos Municipales (CMGRD) de evidencia analítica "
            f"reproducible, eliminando sesgos políticos en la asignación de recursos públicos de prevención."
        )
    elif any(k in q_lower for k in ["memorando", "informe", "reporte", "acta"]):
        reply = (
            f"He preparado el dictamen técnico para **{period}**. Puedes pulsar el botón **'Generar Memorando Ejecutivo Oficial'** o descargarlo "
            f"en formato Markdown (.md) para radicarlo directamente ante la Secretaría del Interior o el Despacho del Gobernador de Santander."
        )
    else:
        reply = (
            f"Como Asistente Técnico del CDGRD Santander, te confirmo que en **{period}** la prioridad departamental está focalizada en "
            f"**{', '.join(muns_names)}**, alcanzando una cobertura de **{total_pob:,.0f} personas expuestas**. "
            f"Los municipios en rotación de descanso son **{cooldown_str}**. ¿Deseas conocer la justificación individual de algún municipio o generar el memorando formal?"
        )

    return reply, offline_source


