"""
Pydantic schemas for API request/response validation.
Compatible with Flutter/Dart models.
"""

from datetime import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, Field


# ==================== Base Schemas ====================

class ResponseBase(BaseModel):
    """Base response schema."""
    success: bool
    message: str


# ==================== Pedagang Schemas ====================

class PedagangBase(BaseModel):
    """Base pedagang schema."""
    nama: str
    username: str


class PedagangCreate(PedagangBase):
    """Schema for creating a new pedagang."""
    password: str


class PedagangResponse(BaseModel):
    """Response schema for pedagang."""
    id: int
    nama: str
    username: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ==================== Authentication Schemas ====================

class LoginRequest(BaseModel):
    """Login request schema."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response schema."""
    success: bool
    message: str
    token: Optional[str] = None
    pedagang: Optional[PedagangResponse] = None


class TokenData(BaseModel):
    """Token data schema."""
    username: Optional[str] = None
    pedagang_id: Optional[int] = None


# ==================== Lokasi Schemas ====================

class LokasiBase(BaseModel):
    """Base lokasi schema."""
    nama: str
    latitude: float
    longitude: float


class LokasiCreate(LokasiBase):
    """Schema for creating a new lokasi."""
    pass


class LokasiUpdate(LokasiBase):
    """Schema for updating a lokasi."""
    pass


class LokasiResponse(BaseModel):
    """Response schema for lokasi."""
    id: int
    pedagang_id: int
    nama: str
    latitude: float
    longitude: float
    created_at: datetime
    
    class Config:
        from_attributes = True


class LokasiListResponse(ResponseBase):
    """Response schema for list of lokasi."""
    data: List[LokasiResponse] = []


# ==================== Kunjungan Schemas ====================

class KunjunganResponse(BaseModel):
    """Response schema untuk satu sesi kunjungan/mangkal."""
    id: int
    pedagang_id: int
    lokasi_id: int
    waktu_mulai: datetime
    waktu_selesai: Optional[datetime] = None
    durasi_mangkal: Optional[float] = None   # menit, None jika masih aktif
    kondisi_cuaca: str
    hari_kuliah: int
    sedang_aktif: bool = False             # True jika waktu_selesai is None
    total_pendapatan: int = 0

    class Config:
        from_attributes = True


class KunjunganDetailResponse(KunjunganResponse):
    """Response schema kunjungan dengan detail lokasi."""
    lokasi: Optional[LokasiResponse] = None


class KunjunganListResponse(ResponseBase):
    """Response schema untuk list kunjungan."""
    data: List[KunjunganResponse] = []


class KunjunganAktifResponse(ResponseBase):
    """Response schema untuk kunjungan yang sedang aktif."""
    data: Optional[KunjunganDetailResponse] = None


# ==================== Transaksi Schemas (Input per Konsumen) ====================

class TransaksiCreate(BaseModel):
    """
    Schema untuk mencatat 1 transaksi per konsumen.
    Backend secara otomatis menentukan / membuat Kunjungan yang tepat.
    """
    lokasi_id: int = Field(..., description="ID lokasi tempat transaksi terjadi")
    jumlah_terjual: int = Field(..., ge=1, description="Jumlah item terjual ke konsumen ini")
    kondisi_cuaca: Optional[str] = Field(
        default="cerah",
        description="Kondisi cuaca: cerah, mendung, hujan"
    )
    hari_kuliah: Optional[int] = Field(
        default=1,
        description="1 = hari kuliah aktif, 0 = akhir pekan / libur"
    )


class TransaksiResponse(BaseModel):
    """Response setelah transaksi berhasil dicatat."""
    id: int
    kunjungan_id: int
    lokasi_id: int
    jumlah_terjual: int
    waktu_transaksi: datetime

    class Config:
        from_attributes = True


class TransaksiCreateResponse(ResponseBase):
    """Response lengkap saat transaksi dicatat, termasuk info kunjungan."""
    transaksi: Optional[TransaksiResponse] = None
    kunjungan: Optional[KunjunganResponse] = None
    kunjungan_baru: bool = False   # True jika kunjungan baru dibuat
    lokasi_berganti: bool = False  # True jika berpindah dari lokasi sebelumnya


# ==================== Penjualan Schemas (Backward Compat + Agregat) ====================

class PenjualanBase(BaseModel):
    """Base penjualan schema."""
    lokasi_id: int
    jumlah_terjual: int = Field(..., ge=0, description="Jumlah item terjual")


class PenjualanCreate(PenjualanBase):
    """Schema for creating a new penjualan record (legacy endpoint)."""
    kondisi_cuaca: Optional[str] = Field(default="cerah", description="Kondisi cuaca: cerah, mendung, hujan")
    hari_kuliah: Optional[int] = Field(default=1, description="1 = hari kuliah, 0 = akhir pekan")


class PenjualanResponse(BaseModel):
    """Response schema for penjualan."""
    id: int
    pedagang_id: int
    lokasi_id: int
    kunjungan_id: Optional[int] = None
    waktu_kunjungan: datetime
    waktu_transaksi: Optional[datetime] = None
    jumlah_terjual: int
    durasi_mangkal: float
    kondisi_cuaca: str
    hari_kuliah: int
    
    class Config:
        from_attributes = True


class PenjualanListResponse(ResponseBase):
    """Response schema for list of penjualan."""
    data: List[PenjualanResponse] = []


class PenjualanCreateResponse(ResponseBase):
    """Response schema for creating penjualan."""
    data: Optional[PenjualanResponse] = None


# ==================== Sesi Penjualan Schemas ====================

class SesiPenjualanStartRequest(BaseModel):
    """Request untuk memulai sesi berjualan."""
    kondisi_cuaca: Optional[str] = Field(
        default="cerah",
        description="Kondisi cuaca: cerah, mendung, hujan"
    )
    hari_kuliah: Optional[int] = Field(
        default=1,
        description="1 = hari kuliah, 0 = akhir pekan"
    )
    stok_awal: Optional[int] = Field(
        default=None,
        description="Jumlah stok tusuk sate awal (opsional)"
    )


class SesiPenjualanResponse(BaseModel):
    """Response schema untuk sesi penjualan."""
    id: int
    pedagang_id: int
    waktu_mulai: datetime
    waktu_selesai: Optional[datetime] = None
    stok_awal: Optional[int] = None
    stok_sisa: Optional[int] = None
    total_pendapatan: int = 0
    total_transaksi: int = 0
    total_lokasi_dikunjungi: int = 0
    durasi_total: Optional[float] = None
    kondisi_cuaca: str = "cerah"
    hari_kuliah: int = 1
    sedang_aktif: bool = False

    class Config:
        from_attributes = True


class SesiAktifResponse(ResponseBase):
    """Response untuk cek sesi aktif."""
    data: Optional[SesiPenjualanResponse] = None
    optimasi: Optional[Any] = None  # Gunakan Any atau dict untuk menghindari cyclic/forward ref issues


class SesiRingkasanResponse(ResponseBase):
    """Response ringkasan setelah sesi ditutup."""
    data: Optional[SesiPenjualanResponse] = None
    kunjungan_list: List[KunjunganResponse] = []


# ==================== Optimasi Schemas ====================

class PerformaHarian(BaseModel):
    """Schema for daily performance. jumlah_terjual = total nominal belanja (rupiah)."""
    tanggal: str
    total_pendapatan: int        # jumlah total nominal belanja semua konsumen per hari
    total_transaksi: int         # jumlah transaksi (konsumen) per hari


class PerformaPenjualanResponse(ResponseBase):
    """Response schema for performance."""
    data: List[PerformaHarian] = []


class OptimasiRequest(BaseModel):
    """Request schema for optimization."""
    kondisi_cuaca: Optional[str] = Field(default="cerah", description="Kondisi cuaca: cerah, mendung, hujan")
    hari_kuliah: Optional[int] = Field(default=1, description="1 = hari kuliah, 0 = akhir pekan")
    max_episodes: Optional[int] = Field(default=100, ge=10, le=1000, description="Jumlah episode training")


class RuteLokasiResponse(BaseModel):
    """Response schema for a location in optimal route."""
    urutan: int
    lokasi_id: int
    nama: str
    latitude: float
    longitude: float
    jarak_dari_sebelumnya: float = 0.0


class DurasiLokasiResponse(BaseModel):
    """Durasi rekomendasi mangkal + reward per lokasi dari hasil Q-Learning."""
    lokasi_id: int
    nama: str
    durasi_menit: int  # Discretized: 30, 60, 120, atau 240
    reward: float = 0.0  # Reward (penjualan) per lokasi


class OptimasiResponse(BaseModel):
    """Response schema for optimization result."""
    success: bool
    message: str

    # ---- Field baru (frontend-friendly) ----
    rekomendasi_rute: List[RuteLokasiResponse] = []
    total_jarak_km: float = 0.0
    perkiraan_penghasilan: int = 0
    kategori: str = ""
    penjelasan: str = ""
    kondisi_cuaca: str = "cerah"
    hari_kuliah: int = 1

    # ---- Field baru: durasi rekomendasi per lokasi ----
    durasi_rekomendasi: List[DurasiLokasiResponse] = []
    total_reward_lokasi: float = 0.0  # Akumulasi reward dari semua lokasi

    # ---- Field lama (backward compat) ----
    rute_optimal: List[RuteLokasiResponse] = []
    total_reward: float = 0.0
    total_jarak: float = 0.0
    rekomendasi: str = ""
    episode_rewards: List[float] = []


# ==================== Episode Schemas ====================

class EpisodeResponse(BaseModel):
    """Response schema for episode."""
    id: int
    pedagang_id: int
    total_reward: float
    total_waktu: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class EpisodeListResponse(ResponseBase):
    """Response schema for list of episodes."""
    data: List[EpisodeResponse] = []


# ==================== Rute Optimal Schemas ====================

class RuteOptimalResponse(BaseModel):
    """Response schema for optimal route."""
    id: int
    pedagang_id: int
    urutan_lokasi: List[int]
    total_jarak: float
    total_reward: float
    rekomendasi: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class RuteOptimalDetailResponse(ResponseBase):
    """Detailed response for optimal route."""
    data: Optional[RuteOptimalResponse] = None
    lokasi_details: List[LokasiResponse] = []
