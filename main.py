from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import httpx
import json
import asyncio
import os

app = FastAPI()

# Securely load API keys from cloud environment variables
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

class Waypoint(BaseModel):
    lat: float
    lon: float

class RouteAnalysisRequest(BaseModel):
    waypoints: List[Waypoint]

@app.post("/api/evaluate-corridor")
async def evaluate_corridor(req: RouteAnalysisRequest):
    total_pts = len(req.waypoints)
    if total_pts < 2:
        return {"error": "Insufficient route data"}
        
    indices = [0, total_pts // 4, total_pts // 2, (3 * total_pts) // 4, total_pts - 1]
    checkpoints = [req.waypoints[i] for i in indices]

    telemetry = []
    max_rain = 0.0
    min_vis = 100.0
    max_wind = 0.0

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        for i, pt in enumerate(checkpoints):
            url = f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={pt.lat},{pt.lon}"
            tasks.append(client.get(url))
            
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, res in enumerate(responses):
            if isinstance(res, httpx.Response) and res.status_code == 200:
                data = res.json().get("current", {})
                rain = data.get("precip_mm", 0.0)
                wind = data.get("wind_kph", 0.0)
                vis = data.get("vis_km", 10.0)
                condition = data.get("condition", {}).get("text", "Clear")
                
                telemetry.append({
                    "segment": f"Point {i+1}",
                    "rain_mm": rain,
                    "wind_kph": wind,
                    "vis_km": vis,
                    "condition": condition
                })
                
                max_rain = max(max_rain, rain)
                max_wind = max(max_wind, wind)
                min_vis = min(min_vis, vis)

    rain_risk = min(max_rain * 12, 50)
    wind_risk = min(max_wind * 0.8, 25)
    vis_risk = 25 if min_vis < 2.0 else (10 if min_vis < 5.0 else 0)
    
    risk_score = min(rain_risk + wind_risk + vis_risk, 100)
    is_safe = risk_score < 45

    if not telemetry:
        return {"error": "Telemetry failure"}

    prompt = (
        f"RainRoute Multi-Variable Telemetry:\n"
        f"Risk Score: {risk_score:.0f}/100\n"
        f"Extremes -> Max Rain: {max_rain}mm, Max Wind: {max_wind}kph, Min Vis: {min_vis}km\n"
        f"Segment Breakdown: {telemetry}\n\n"
        f"Output ONLY a valid JSON object with exactly these keys: "
        f"'threat_level' (LOW, MODERATE, SEVERE, or CRITICAL), "
        f"'primary_hazard', 'choke_point_advisory', 'speed_recommendation', 'detour_advice'."
    )
    
    try:
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            ai_res = await client.post(groq_url, headers=headers, json=payload)
            ai_json = ai_res.json()
            
            if "choices" not in ai_json:
                raise Exception("Groq rejected the API key or request.")
            
            raw_text = ai_json["choices"][0]["message"]["content"]
            ai_data = json.loads(raw_text)
            
    except Exception as e:
        ai_data = {
            "threat_level": "SEVERE" if risk_score > 60 else ("MODERATE" if risk_score > 30 else "LOW"),
            "primary_hazard": "High Winds & Rain" if max_rain > 0 else "Standard Conditions",
            "choke_point_advisory": f"Peak exposure at highest wind point ({max_wind} kph).",
            "speed_recommendation": "Reduce speed by 15%." if risk_score > 30 else "Normal speeds permitted.",
            "detour_advice": "Proceed with caution on primary green route."
        }

    return {
        "telemetry": telemetry,
        "max_rain_mm": max_rain,
        "min_vis_km": min_vis,
        "max_wind_kph": max_wind,
        "risk_score": round(risk_score, 1),
        "is_safe": is_safe,
        "ai_guidance": ai_data
    }

@app.get("/")
def get_dashboard():
    return FileResponse("index.html")