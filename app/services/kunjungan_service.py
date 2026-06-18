"""
Kunjungan Service — Logika bisnis utama untuk sistem tracking pedagang.

Mengelola:
- Sesi mangkal (Kunjungan) per lokasi
- Transisi antar lokasi dalam satu hari
- Pencatatan transaksi per konsumen (Penjualan)
- Penghitungan durasi mangkal otomatis
"""

from datetime import datetime
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import Kunjungan, Penjualan, Lokasi, SesiPenjualan


# =========================================================
# KUNJUNGAN — Fungsi-fungsi dasar
# =========================================================

def get_active_kunjungan(db: Session, pedagang_id: int) -> Optional[Kunjungan]:
    """
    Mencari kunjungan yang sedang aktif (waktu_selesai IS NULL) untuk pedagang.
    Satu pedagang hanya boleh memiliki 1 kunjungan aktif dalam satu waktu.

    Returns:
        Kunjungan aktif jika ada, None jika tidak ada.
    """
    return db.query(Kunjungan).filter(
        Kunjungan.pedagang_id == pedagang_id,
        Kunjungan.waktu_selesai == None  # noqa: E711
    ).order_by(
        Kunjungan.waktu_mulai.desc()
    ).first()


def create_kunjungan(
    db: Session,
    pedagang_id: int,
    lokasi_id: int,
    kondisi_cuaca: str = "cerah",
    hari_kuliah: int = 1,
    sesi_id: Optional[int] = None
) -> Kunjungan:
    """
    Membuat kunjungan baru (session mangkal baru di lokasi tertentu).
    waktu_selesai dibiarkan NULL — artinya kunjungan sedang berlangsung.

    Returns:
        Objek Kunjungan yang baru dibuat.
    """
    kunjungan = Kunjungan(
        pedagang_id=pedagang_id,
        lokasi_id=lokasi_id,
        sesi_id=sesi_id,
        waktu_mulai=datetime.utcnow(),
        waktu_selesai=None,
        durasi_mangkal=None,
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah
    )
    db.add(kunjungan)
    db.commit()
    db.refresh(kunjungan)
    return kunjungan


def close_kunjungan(db: Session, kunjungan: Kunjungan) -> Kunjungan:
    """
    Menutup kunjungan yang sedang aktif:
    - Set waktu_selesai = sekarang
    - Hitung durasi_mangkal (menit) dari selisih waktu_mulai dan waktu_selesai
    - Update record penjualan agregat untuk Q-Learning (jika ada)

    Args:
        kunjungan: Objek Kunjungan yang akan ditutup (harus aktif / waktu_selesai None)

    Returns:
        Kunjungan yang sudah ditutup dengan durasi_mangkal terisi.
    """
    now = datetime.utcnow()
    kunjungan.waktu_selesai = now

    # Hitung durasi dalam MENIT (float), minimum 1 menit
    delta = now - kunjungan.waktu_mulai
    durasi_menit = round(delta.total_seconds() / 60, 2)
    kunjungan.durasi_mangkal = max(1.0, durasi_menit)

    # Update durasi_mangkal di record penjualan agregat (backward compat Q-Learning)
    _sync_durasi_to_penjualan_agregat(db, kunjungan, durasi_menit)

    db.commit()
    db.refresh(kunjungan)
    return kunjungan


def _sync_durasi_to_penjualan_agregat(
    db: Session, kunjungan: Kunjungan, durasi_menit: float
):
    """
    Sinkronisasi durasi mangkal (menit) ke record penjualan agregat (tabel lama).
    Dipanggil saat kunjungan ditutup agar Q-Learning mendapatkan data durasi akurat.
    """
    agregat = db.query(Penjualan).filter(
        Penjualan.pedagang_id == kunjungan.pedagang_id,
        Penjualan.lokasi_id == kunjungan.lokasi_id,
        Penjualan.kunjungan_id == kunjungan.id
    ).first()

    if agregat:
        agregat.durasi_mangkal = durasi_menit
        # Tidak di-commit di sini — akan di-commit oleh pemanggil (close_kunjungan)


# =========================================================
# PENJUALAN — Fungsi-fungsi transaksi
# =========================================================

def create_penjualan(
    db: Session,
    kunjungan: Kunjungan,
    jumlah_terjual: int
) -> Penjualan:
    """
    Mencatat 1 transaksi individual per konsumen ke tabel penjualan.
    Transaksi ini terhubung ke kunjungan aktif via kunjungan_id.

    Returns:
        Objek Penjualan yang baru dicatat.
    """
    now = datetime.utcnow()
    penjualan = Penjualan(
        pedagang_id=kunjungan.pedagang_id,
        lokasi_id=kunjungan.lokasi_id,
        kunjungan_id=kunjungan.id,
        jumlah_terjual=jumlah_terjual,
        waktu_transaksi=now,
        waktu_kunjungan=now,              # backward compat
        kondisi_cuaca=kunjungan.kondisi_cuaca,
        hari_kuliah=kunjungan.hari_kuliah,
        durasi_mangkal=0                  # diperbarui nanti saat close_kunjungan
    )
    db.add(penjualan)
    db.commit()
    db.refresh(penjualan)
    return penjualan


# =========================================================
# PROCESS TRANSAKSI — Orkestrasi utama (entry point)
# =========================================================

def process_transaksi(
    db: Session,
    pedagang_id: int,
    lokasi_id: int,
    jumlah_terjual: int,
    kondisi_cuaca: str = "cerah",
    hari_kuliah: int = 1
) -> dict:
    """
    Fungsi orkestrasi utama untuk memproses satu transaksi per konsumen.

    Logika yang dijalankan:
    1. Cek apakah ada kunjungan aktif (waktu_selesai IS NULL)
       a. Tidak ada → buat kunjungan baru di lokasi ini
       b. Ada, lokasi BERBEDA → tutup kunjungan lama, buat kunjungan baru
       c. Ada, lokasi SAMA → gunakan kunjungan yang ada
    2. Catat transaksi ke tabel penjualan

    Returns:
        dict dengan keys:
        - transaksi: Penjualan object
        - kunjungan: Kunjungan object (aktif)
        - kunjungan_baru: bool — True jika kunjungan baru dibuat
        - lokasi_berganti: bool — True jika pindah dari lokasi sebelumnya
    """
    kunjungan_baru = False
    lokasi_berganti = False

    # Cari sesi aktif untuk di-link ke kunjungan
    active_sesi = db.query(SesiPenjualan).filter(
        SesiPenjualan.pedagang_id == pedagang_id,
        SesiPenjualan.waktu_selesai == None  # noqa: E711
    ).first()
    sesi_id = active_sesi.id if active_sesi else None

    # 1. Cek kunjungan aktif
    active = get_active_kunjungan(db, pedagang_id)

    if active is None:
        # Skenario A: Tidak ada kunjungan aktif → buat baru
        kunjungan = create_kunjungan(db, pedagang_id, lokasi_id, kondisi_cuaca, hari_kuliah, sesi_id=sesi_id)
        kunjungan_baru = True

    elif active.lokasi_id != lokasi_id:
        # Skenario B: Ada kunjungan aktif tapi di lokasi berbeda
        # → Tutup kunjungan lama, buat kunjungan baru
        close_kunjungan(db, active)
        kunjungan = create_kunjungan(db, pedagang_id, lokasi_id, kondisi_cuaca, hari_kuliah, sesi_id=sesi_id)
        kunjungan_baru = True
        lokasi_berganti = True

    else:
        # Skenario C: Ada kunjungan aktif di lokasi yang sama → pakai yang ada
        kunjungan = active

    # 2. Catat transaksi
    transaksi = create_penjualan(db, kunjungan, jumlah_terjual)

    return {
        "transaksi": transaksi,
        "kunjungan": kunjungan,
        "kunjungan_baru": kunjungan_baru,
        "lokasi_berganti": lokasi_berganti
    }


# =========================================================
# QUERY HELPERS
# =========================================================

def get_kunjungan_list(
    db: Session,
    pedagang_id: int,
    limit: int = 20,
    offset: int = 0,
    lokasi_id: Optional[int] = None,
    hanya_selesai: bool = False,
    hanya_aktif: bool = False
):
    """
    Mengambil list kunjungan dengan berbagai filter.

    Args:
        hanya_selesai: Jika True, hanya tampilkan kunjungan yang sudah ditutup
        hanya_aktif: Jika True, hanya tampilkan kunjungan yang masih berlangsung
    """
    query = db.query(Kunjungan).filter(
        Kunjungan.pedagang_id == pedagang_id
    )

    if lokasi_id is not None:
        query = query.filter(Kunjungan.lokasi_id == lokasi_id)

    if hanya_selesai:
        query = query.filter(Kunjungan.waktu_selesai != None)  # noqa: E711

    if hanya_aktif:
        query = query.filter(Kunjungan.waktu_selesai == None)  # noqa: E711

    return query.order_by(
        Kunjungan.waktu_mulai.desc()
    ).offset(offset).limit(limit).all()


def get_transaksi_by_kunjungan(
    db: Session,
    kunjungan_id: int
) -> list:
    """Mengambil semua transaksi individual dalam satu kunjungan."""
    return db.query(Penjualan).filter(
        Penjualan.kunjungan_id == kunjungan_id
    ).order_by(Penjualan.waktu_transaksi.asc()).all()
