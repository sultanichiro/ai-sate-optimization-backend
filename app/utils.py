import json
import urllib.request
import urllib.error
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def get_cuaca_otomatis(lat: float, lon: float) -> str:
    """
    Mengambil cuaca saat ini dari Open-Meteo berdasarkan latitude dan longitude.
    Mapping cuaca dari WMO weather code:
    - 0-3: cerah (Clear, Partly cloudy)
    - 45-48: mendung (Fog, Rime fog)
    - 51-99: hujan (Drizzle, Rain, Snow, Thunderstorm)
    
    Default jika gagal adalah 'cerah'.
    """
    # Open-Meteo API tidak memerlukan API Key
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if 'current_weather' in data and 'weathercode' in data['current_weather']:
                    code = data['current_weather']['weathercode']
                    if code <= 3:
                        return "cerah"
                    elif code <= 48:
                        return "mendung"
                    else:
                        return "hujan"
    except Exception as e:
        logger.error(f"Gagal mengambil cuaca dari Open-Meteo: {e}")
    
    return "cerah"  # Fallback

def get_hari_otomatis() -> int:
    """
    Menentukan apakah hari ini hari kuliah atau hari libur.
    Senin-Jumat = 1 (hari kuliah)
    Sabtu-Minggu = 0 (hari libur)
    """
    now = datetime.now()
    # weekday(): Senin = 0, Selasa = 1, ..., Minggu = 6
    if now.weekday() < 5:
        return 1
    return 0
