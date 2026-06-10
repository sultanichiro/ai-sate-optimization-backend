"""
Reward Service — Konversi reward Q-Learning menjadi informasi
yang mudah dipahami oleh pedagang (Perkiraan Penghasilan, Kategori, Penjelasan).

Modul ini bersifat stateless dan dapat di-import di mana saja.
"""

from typing import Dict, List, Optional


# ==================== Konstanta Kategori ====================

KATEGORI_SEPI = "Sepi"
KATEGORI_LUMAYAN = "Lumayan"
KATEGORI_RAMAI = "Ramai"
KATEGORI_SANGAT_MENGUNTUNGKAN = "Sangat Menguntungkan"

BATAS_SEPI = 200_000
BATAS_LUMAYAN = 400_000
BATAS_RAMAI = 800_000


# ==================== Modifier Cuaca & Hari ====================

CUACA_MODIFIER = {
    "cerah": 1.0,
    "mendung": 0.8,
    "hujan": 0.5,
}

HARI_MODIFIER_KULIAH = 1.2   # hari kuliah aktif → lebih ramai
HARI_MODIFIER_WEEKEND = 0.7  # akhir pekan / libur → lebih sepi


def hitung_modifier(kondisi_cuaca: str = "cerah", hari_kuliah: int = 1) -> float:
    """
    Menghitung multiplier gabungan berdasarkan cuaca dan tipe hari.

    Args:
        kondisi_cuaca: "cerah" | "mendung" | "hujan"
        hari_kuliah: 1 = hari kuliah aktif, 0 = akhir pekan / libur

    Returns:
        Multiplier float (contoh: 1.0 × 1.2 = 1.2 untuk cerah + hari kuliah)
    """
    cuaca_mod = CUACA_MODIFIER.get(kondisi_cuaca.lower(), 1.0)
    hari_mod = HARI_MODIFIER_KULIAH if hari_kuliah else HARI_MODIFIER_WEEKEND
    return cuaca_mod * hari_mod


# ==================== Konversi Reward → Penghasilan ====================

def hitung_perkiraan_penghasilan(
    lokasi_stats: Dict[int, Dict[str, float]],
    route_lokasi_ids: List[int],
    kondisi_cuaca: str = "cerah",
    hari_kuliah: int = 1,
) -> int:
    """
    Menghitung perkiraan penghasilan HARIAN (Rupiah) berdasarkan data historis
    lokasi-lokasi di rute optimal.

    Logika:
    1. Untuk setiap lokasi di rute, ambil rata-rata jumlah_terjual per kunjungan
       (sudah Rupiah — nominal belanja).
    2. Jumlahkan semua lokasi → ini adalah estimasi pendapatan 1 hari
       jika pedagang mengunjungi semua lokasi di rute.
    3. Kalikan dengan modifier cuaca × hari.

    Args:
        lokasi_stats: Dict statistik per lokasi dari QLearningEnvironment
                      {lokasi_id: {"avg_terjual": float, "total_terjual": float, ...}}
        route_lokasi_ids: List ID lokasi di rute optimal (urutan kunjungan)
        kondisi_cuaca: Kondisi cuaca saat ini
        hari_kuliah: 1 = hari kuliah, 0 = weekend

    Returns:
        Perkiraan penghasilan harian dalam Rupiah (int, dibulatkan)
    """
    if not route_lokasi_ids or not lokasi_stats:
        return 0

    # Jumlahkan rata-rata pendapatan dari setiap lokasi di rute
    # Ini merepresentasikan total pendapatan 1 hari jika pedagang
    # mengunjungi semua lokasi di rute
    total_penghasilan_harian = 0.0

    for lokasi_id in route_lokasi_ids:
        stats = lokasi_stats.get(lokasi_id, {})
        avg_terjual = stats.get("avg_terjual", 0.0)
        total_penghasilan_harian += avg_terjual

    # Terapkan modifier cuaca & hari
    modifier = hitung_modifier(kondisi_cuaca, hari_kuliah)
    total_penghasilan_harian *= modifier

    return int(round(total_penghasilan_harian))


# ==================== Kategori Penghasilan ====================

def tentukan_kategori(penghasilan: int) -> str:
    """
    Menentukan kategori berdasarkan nilai perkiraan penghasilan.

    Threshold:
        < 1.000.000          → "Sepi"
        1.000.000 – 3.000.000 → "Lumayan"
        3.000.000 – 5.000.000 → "Ramai"
        > 5.000.000          → "Sangat Menguntungkan"

    Args:
        penghasilan: Perkiraan penghasilan dalam Rupiah

    Returns:
        String kategori
    """
    if penghasilan < BATAS_SEPI:
        return KATEGORI_SEPI
    elif penghasilan < BATAS_LUMAYAN:
        return KATEGORI_LUMAYAN
    elif penghasilan < BATAS_RAMAI:
        return KATEGORI_RAMAI
    else:
        return KATEGORI_SANGAT_MENGUNTUNGKAN


# ==================== Penjelasan Otomatis ====================

def _format_rupiah(nilai: int) -> str:
    """Format angka menjadi string Rupiah yang mudah dibaca."""
    if nilai >= 1_000_000:
        juta = nilai / 1_000_000
        if juta == int(juta):
            return f"Rp {int(juta)} juta"
        return f"Rp {juta:,.1f} juta".replace(",", ".")
    elif nilai >= 1_000:
        ribu = nilai / 1_000
        if ribu == int(ribu):
            return f"Rp {int(ribu)} ribu"
        return f"Rp {ribu:,.1f} ribu".replace(",", ".")
    return f"Rp {nilai:,}".replace(",", ".")


def buat_penjelasan(
    kategori: str,
    kondisi_cuaca: str,
    hari_kuliah: int,
    penghasilan: int,
    jumlah_lokasi: int = 0,
) -> str:
    """
    Membuat penjelasan yang mudah dipahami pedagang.

    Args:
        kategori: Hasil dari tentukan_kategori()
        kondisi_cuaca: "cerah" | "mendung" | "hujan"
        hari_kuliah: 1 = hari kuliah, 0 = weekend
        penghasilan: Perkiraan penghasilan (Rupiah)
        jumlah_lokasi: Jumlah lokasi di rute

    Returns:
        String penjelasan dalam bahasa Indonesia
    """
    # Deskripsi cuaca
    cuaca_desc = {
        "cerah": "cuaca cerah",
        "mendung": "cuaca mendung",
        "hujan": "cuaca hujan",
    }.get(kondisi_cuaca.lower(), "cuaca tidak diketahui")

    # Deskripsi hari
    hari_desc = "hari kuliah aktif" if hari_kuliah else "akhir pekan"

    # Dampak cuaca pada penjelasan
    cuaca_dampak = {
        "cerah": "Kondisi cuaca cerah mendukung penjualan optimal.",
        "mendung": "Cuaca mendung dapat sedikit mengurangi jumlah pembeli.",
        "hujan": "Cuaca hujan kemungkinan mengurangi jumlah pembeli secara signifikan.",
    }.get(kondisi_cuaca.lower(), "")

    # Dampak hari
    if hari_kuliah:
        hari_dampak = "Hari kuliah aktif biasanya lebih ramai."
    else:
        hari_dampak = "Akhir pekan biasanya lebih sepi dibanding hari kuliah."

    # Komposisi penjelasan
    rupiah_str = _format_rupiah(penghasilan)
    rute_info = f" melalui {jumlah_lokasi} lokasi" if jumlah_lokasi > 0 else ""

    penjelasan = (
        f"Perkiraan penghasilan sekitar {rupiah_str}{rute_info} "
        f"saat {cuaca_desc} dan {hari_desc}. "
        f"{cuaca_dampak} {hari_dampak}"
    ).strip()

    return penjelasan


# ==================== Fungsi Utama (Convenience) ====================

def proses_reward(
    lokasi_stats: Dict[int, Dict[str, float]],
    route_lokasi_ids: List[int],
    kondisi_cuaca: str = "cerah",
    hari_kuliah: int = 1,
) -> Dict:
    """
    Fungsi utama yang menggabungkan seluruh proses konversi reward.

    Returns:
        Dict berisi:
        - perkiraan_penghasilan: int (Rupiah)
        - kategori: str
        - penjelasan: str
        - modifier: float
    """
    penghasilan = hitung_perkiraan_penghasilan(
        lokasi_stats=lokasi_stats,
        route_lokasi_ids=route_lokasi_ids,
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
    )

    kategori = tentukan_kategori(penghasilan)

    penjelasan = buat_penjelasan(
        kategori=kategori,
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
        penghasilan=penghasilan,
        jumlah_lokasi=len(route_lokasi_ids),
    )

    modifier = hitung_modifier(kondisi_cuaca, hari_kuliah)

    return {
        "perkiraan_penghasilan": penghasilan,
        "kategori": kategori,
        "penjelasan": penjelasan,
        "modifier": modifier,
    }
