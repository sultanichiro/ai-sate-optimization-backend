from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    Boolean
)

from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# =========================================================
# PEDAGANG
# =========================================================
class Pedagang(Base):
    __tablename__ = "pedagang"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    nama = Column(String(255), nullable=False)

    username = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )

    password = Column(String(255), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    lokasi = relationship(
        "Lokasi",
        back_populates="pedagang",
        cascade="all, delete-orphan"
    )

    penjualan = relationship(
        "Penjualan",
        back_populates="pedagang",
        cascade="all, delete-orphan"
    )

    kunjungan = relationship(
        "Kunjungan",
        back_populates="pedagang",
        cascade="all, delete-orphan"
    )

    episodes = relationship(
        "Episode",
        back_populates="pedagang",
        cascade="all, delete-orphan"
    )

    qtable_entries = relationship(
        "QTableEntry",
        back_populates="pedagang",
        cascade="all, delete-orphan"
    )

    rute_optimal = relationship(
        "RuteOptimal",
        back_populates="pedagang",
        cascade="all, delete-orphan"
    )

    sesi_penjualan = relationship(
        "SesiPenjualan",
        back_populates="pedagang",
        cascade="all, delete-orphan"
    )


# =========================================================
# LOKASI
# =========================================================
class Lokasi(Base):
    __tablename__ = "lokasi"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    pedagang_id = Column(
        Integer,
        ForeignKey("pedagang.id"),
        nullable=False,
        index=True
    )

    nama = Column(String(255), nullable=False)

    latitude = Column(Float, nullable=False)

    longitude = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pedagang = relationship(
        "Pedagang",
        back_populates="lokasi"
    )

    penjualan = relationship(
        "Penjualan",
        back_populates="lokasi",
        cascade="all, delete-orphan"
    )

    kunjungan = relationship(
        "Kunjungan",
        back_populates="lokasi",
        cascade="all, delete-orphan"
    )


# =========================================================
# KUNJUNGAN (Session Mangkal)
# =========================================================
class Kunjungan(Base):
    """
    Merepresentasikan satu sesi mangkal pedagang di satu lokasi.
    Satu pedagang bisa memiliki banyak kunjungan dalam 1 hari (multi-lokasi).
    waktu_selesai = None berarti kunjungan masih aktif.
    """
    __tablename__ = "kunjungan"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    pedagang_id = Column(
        Integer,
        ForeignKey("pedagang.id"),
        nullable=False,
        index=True
    )

    lokasi_id = Column(
        Integer,
        ForeignKey("lokasi.id"),
        nullable=False,
        index=True
    )

    # FK ke sesi penjualan — nullable untuk backward compat
    sesi_id = Column(
        Integer,
        ForeignKey("sesi_penjualan.id"),
        nullable=True,
        index=True
    )

    waktu_mulai = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    # NULL = kunjungan masih aktif / sedang berlangsung
    waktu_selesai = Column(DateTime, nullable=True)

    # Durasi dalam MENIT (float) — dihitung otomatis saat close_kunjungan()
    # Contoh: 90.0 = 1 jam 30 menit
    durasi_mangkal = Column(Float, nullable=True)

    kondisi_cuaca = Column(
        String(50),
        default="cerah"
    )

    hari_kuliah = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pedagang = relationship(
        "Pedagang",
        back_populates="kunjungan"
    )

    lokasi = relationship(
        "Lokasi",
        back_populates="kunjungan"
    )

    # Penjualan yang terjadi selama kunjungan ini
    transaksi = relationship(
        "Penjualan",
        back_populates="kunjungan",
        foreign_keys="Penjualan.kunjungan_id"
    )

    sesi = relationship(
        "SesiPenjualan",
        back_populates="kunjungan_list"
    )


# =========================================================
# PENJUALAN (Transaksi per Konsumen + Agregat untuk Q-Learning)
# =========================================================
class Penjualan(Base):
    """
    Merepresentasikan transaksi. Bisa berupa:
    - Transaksi individual per konsumen (kunjungan_id NOT NULL)
    - Agregat harian per lokasi untuk Q-Learning (kunjungan_id NULL = legacy)
    """
    __tablename__ = "penjualan"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    pedagang_id = Column(
        Integer,
        ForeignKey("pedagang.id"),
        nullable=False,
        index=True
    )

    lokasi_id = Column(
        Integer,
        ForeignKey("lokasi.id"),
        nullable=False,
        index=True
    )

    # FK ke kunjungan — nullable agar backward compatible dengan data lama
    kunjungan_id = Column(
        Integer,
        ForeignKey("kunjungan.id"),
        nullable=True,
        index=True
    )

    jumlah_terjual = Column(Integer, nullable=False)

    # Waktu transaksi individual (sebelumnya waktu_kunjungan)
    waktu_transaksi = Column(
        DateTime,
        default=datetime.utcnow
    )

    # Dipertahankan untuk backward compat dengan Q-Learning
    waktu_kunjungan = Column(
        DateTime,
        default=datetime.utcnow
    )

    # Durasi dalam MENIT (float) — disinkronisasi dari Kunjungan saat close
    # Backward compat untuk Q-Learning
    durasi_mangkal = Column(Float, nullable=False, default=0.0)

    kondisi_cuaca = Column(
        String(50),
        default="cerah"
    )

    hari_kuliah = Column(Integer, default=1)

    reward = Column(Float, default=0.0)

    # Relationships
    pedagang = relationship(
        "Pedagang",
        back_populates="penjualan"
    )

    lokasi = relationship(
        "Lokasi",
        back_populates="penjualan"
    )

    kunjungan = relationship(
        "Kunjungan",
        back_populates="transaksi",
        foreign_keys=[kunjungan_id]
    )


# =========================================================
# EPISODE
# =========================================================
class Episode(Base):
    __tablename__ = "episode"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    pedagang_id = Column(
        Integer,
        ForeignKey("pedagang.id"),
        nullable=False,
        index=True
    )

    total_reward = Column(Float, default=0.0)

    total_waktu = Column(Integer, default=0)

    jumlah_langkah = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pedagang = relationship(
        "Pedagang",
        back_populates="episodes"
    )


# =========================================================
# Q TABLE
# =========================================================
class QTableEntry(Base):
    __tablename__ = "q_table"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    pedagang_id = Column(
        Integer,
        ForeignKey("pedagang.id"),
        nullable=False,
        index=True
    )

    state_key = Column(
        String(500),
        nullable=False,
        index=True
    )

    action = Column(Integer, nullable=False)

    q_value = Column(Float, default=0.0)

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # Relationships
    pedagang = relationship(
        "Pedagang",
        back_populates="qtable_entries"
    )


# =========================================================
# RUTE OPTIMAL
# =========================================================
class RuteOptimal(Base):
    __tablename__ = "rute_optimal"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    pedagang_id = Column(
        Integer,
        ForeignKey("pedagang.id"),
        nullable=False,
        index=True
    )

    urutan_lokasi = Column(JSON, nullable=False)

    total_jarak = Column(Float, default=0.0)

    total_reward = Column(Float, default=0.0)

    rekomendasi = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pedagang = relationship(
        "Pedagang",
        back_populates="rute_optimal"
    )


# =========================================================
# SESI PENJUALAN (Session Harian Start/Stop)
# =========================================================
class SesiPenjualan(Base):
    """
    Merepresentasikan satu sesi berjualan harian pedagang.
    Dari Start (berangkat/mulai jualan) sampai Stop (pulang/stok habis).
    Satu SesiPenjualan bisa memiliki banyak Kunjungan (multi-lokasi).
    """
    __tablename__ = "sesi_penjualan"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    pedagang_id = Column(
        Integer,
        ForeignKey("pedagang.id"),
        nullable=False,
        index=True
    )

    waktu_mulai = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    # NULL = sesi masih aktif / sedang berjualan
    waktu_selesai = Column(DateTime, nullable=True)

    # Stok awal tusuk sate (opsional)
    stok_awal = Column(Integer, nullable=True)

    # Stok tersisa saat stop (opsional)
    stok_sisa = Column(Integer, nullable=True)

    # Ringkasan — dihitung saat stop
    total_pendapatan = Column(Integer, default=0)
    total_transaksi = Column(Integer, default=0)
    total_lokasi_dikunjungi = Column(Integer, default=0)
    durasi_total = Column(Float, nullable=True)  # Menit

    kondisi_cuaca = Column(String(50), default="cerah")
    hari_kuliah = Column(Integer, default=1)

    catatan = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pedagang = relationship(
        "Pedagang",
        back_populates="sesi_penjualan"
    )

    kunjungan_list = relationship(
        "Kunjungan",
        back_populates="sesi",
        cascade="all, delete-orphan"
    )