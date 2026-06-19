"""
FastAPI Main Application for Sate Keliling Route Optimization.
Implements all API endpoints for Flutter integration.
"""

from datetime import timedelta
from typing import List
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
# from app.controller.sistem_controller import SistemController
from app.database import db_manager
from app.models import Pedagang, Lokasi, Penjualan
from app.schemas import (
    # Auth
    LoginRequest, LoginResponse, PedagangCreate, PedagangResponse,
    # Lokasi
    LokasiCreate, LokasiUpdate, LokasiResponse, LokasiListResponse,
    # Penjualan (legacy)
    PenjualanCreate, PenjualanResponse, PenjualanListResponse, PenjualanCreateResponse, PerformaPenjualanResponse, PerformaHarian,
    # Kunjungan
    KunjunganResponse, KunjunganDetailResponse, KunjunganListResponse, KunjunganAktifResponse, PindahLokasiRequest,
    # Transaksi
    TransaksiCreate, TransaksiResponse, TransaksiCreateResponse,
    # Sesi Penjualan
    SesiPenjualanStartRequest, SesiPenjualanResponse, SesiAktifResponse, SesiRingkasanResponse,
    # Optimasi
    OptimasiRequest, OptimasiResponse, RuteLokasiResponse, DurasiLokasiResponse,
    RekomendasiSelanjutnyaRequest, RekomendasiSelanjutnyaResponse,
    # Episode
    EpisodeListResponse, EpisodeResponse,
    # Rute Optimal
    RuteOptimalDetailResponse, RuteOptimalResponse,
    # Base
    ResponseBase
)
from app.auth import (
    get_password_hash, create_access_token, authenticate_pedagang, get_current_pedagang
)
from app.agent import SistemController
from app.services import kunjungan_service
from app.services.reward_service import proses_reward
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.utils import get_cuaca_otomatis, get_hari_otomatis

# ==================== FastAPI App ====================

app = FastAPI(
    title="Sate Keliling Optimization API",
    description="Backend API untuk Optimasi Alokasi Waktu dan Penentuan Prioritas Singgah Pedagang Sate Keliling menggunakan Q-Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# ==================== Dependency DB ====================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# ==================== Startup Event ====================

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    db_manager.create_tables()
    print("Database tables created/verified")


# ==================== Root Endpoint ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information."""
    return {
        "message": "Selamat datang di API Optimasi Sate Keliling",
        "version": "1.0.0",
        "docs": "/docs"
    }


# ==================== Authentication Endpoints ====================

@app.post("/login", response_model=LoginResponse, tags=["Authentication"])
async def login(
    request: LoginRequest,
    db: Session = Depends(db_manager.get_db)
):
    """
    Autentikasi pedagang.
    
    Args:
        request: LoginRequest dengan username dan password
        
    Returns:
        LoginResponse dengan token JWT jika berhasil
    """
    pedagang = authenticate_pedagang(db, request.username, request.password)
    
    if not pedagang:
        return LoginResponse(
            success=False,
            message="Username atau password salah",
            token=None,
            pedagang=None
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": pedagang.username, "pedagang_id": pedagang.id},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return LoginResponse(
        success=True,
        message="Login berhasil",
        token=access_token,
        pedagang=PedagangResponse(
            id=pedagang.id,
            nama=pedagang.nama,
            username=pedagang.username,
            created_at=pedagang.created_at
        )
    )


@app.post("/register", response_model=LoginResponse, tags=["Authentication"])
async def register(
    request: PedagangCreate,
    db: Session = Depends(db_manager.get_db)
):
    """
    Registrasi pedagang baru.
    
    Args:
        request: PedagangCreate dengan nama, username, dan password
        
    Returns:
        LoginResponse dengan token JWT jika berhasil
    """
    # Check if username already exists
    existing = db_manager.get_pedagang_by_username(db, request.username)
    if existing:
        return LoginResponse(
            success=False,
            message="Username sudah digunakan",
            token=None,
            pedagang=None
        )
    
    # Create new pedagang
    hashed_password = get_password_hash(request.password)
    pedagang = db_manager.create_pedagang(
        db, request.nama, request.username, hashed_password
    )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": pedagang.username, "pedagang_id": pedagang.id},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return LoginResponse(
        success=True,
        message="Registrasi berhasil",
        token=access_token,
        pedagang=PedagangResponse(
            id=pedagang.id,
            nama=pedagang.nama,
            username=pedagang.username,
            created_at=pedagang.created_at
        )
    )


@app.post("/refresh-token", response_model=LoginResponse, tags=["Authentication"])
async def refresh_token(
    pedagang: Pedagang = Depends(get_current_pedagang)
):
    """
    Memperbarui token JWT.
    
    Returns:
        LoginResponse dengan token JWT baru jika berhasil
    """
    # Create access token
    access_token = create_access_token(
        data={"sub": pedagang.username, "pedagang_id": pedagang.id},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return LoginResponse(
        success=True,
        message="Token berhasil diperbarui",
        token=access_token,
        pedagang=PedagangResponse(
            id=pedagang.id,
            nama=pedagang.nama,
            username=pedagang.username,
            created_at=pedagang.created_at
        )
    )


# ==================== Lokasi Endpoints ====================

@app.get("/lokasi", response_model=LokasiListResponse, tags=["Lokasi"])
async def get_lokasi(
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Mendapatkan semua titik mangkal pedagang.
    
    Returns:
        LokasiListResponse dengan daftar lokasi
    """
    lokasi_list = db_manager.get_lokasi(db, pedagang.id)
    
    return LokasiListResponse(
        success=True,
        message=f"Ditemukan {len(lokasi_list)} lokasi",
        data=[LokasiResponse(
            id=l.id,
            pedagang_id=l.pedagang_id,
            nama=l.nama,
            latitude=l.latitude,
            longitude=l.longitude,
            created_at=l.created_at
        ) for l in lokasi_list]
    )


@app.post("/lokasi", response_model=LokasiListResponse, tags=["Lokasi"])
async def create_lokasi(
    request: LokasiCreate,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Menambahkan titik mangkal baru.
    
    Args:
        request: LokasiCreate dengan nama, latitude, longitude
        
    Returns:
        LokasiListResponse dengan lokasi baru
    """
    lokasi = db_manager.create_lokasi(
        db, pedagang.id, request.nama, request.latitude, request.longitude
    )
    
    return LokasiListResponse(
        success=True,
        message=f"Lokasi '{lokasi.nama}' berhasil ditambahkan",
        data=[LokasiResponse(
            id=lokasi.id,
            pedagang_id=lokasi.pedagang_id,
            nama=lokasi.nama,
            latitude=lokasi.latitude,
            longitude=lokasi.longitude,
            created_at=lokasi.created_at
        )]
    )


@app.put("/lokasi/{lokasi_id}", response_model=LokasiListResponse, tags=["Lokasi"])
async def update_lokasi(
    lokasi_id: int,
    request: LokasiUpdate,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Mengupdate titik mangkal.
    
    Args:
        lokasi_id: ID lokasi yang akan diupdate
        request: LokasiUpdate dengan data baru
        
    Returns:
        LokasiListResponse dengan lokasi yang diupdate
    """
    # Check if lokasi exists and belongs to pedagang
    lokasi = db_manager.get_lokasi_by_id(db, lokasi_id)
    if not lokasi or lokasi.pedagang_id != pedagang.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lokasi tidak ditemukan"
        )
    
    updated = db_manager.update_lokasi(
        db, lokasi_id, request.nama, request.latitude, request.longitude
    )
    
    return LokasiListResponse(
        success=True,
        message=f"Lokasi '{updated.nama}' berhasil diupdate",
        data=[LokasiResponse(
            id=updated.id,
            pedagang_id=updated.pedagang_id,
            nama=updated.nama,
            latitude=updated.latitude,
            longitude=updated.longitude,
            created_at=updated.created_at
        )]
    )


@app.delete("/lokasi/{lokasi_id}", response_model=ResponseBase, tags=["Lokasi"])
async def delete_lokasi(
    lokasi_id: int,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Menghapus titik mangkal.
    
    Args:
        lokasi_id: ID lokasi yang akan dihapus
        
    Returns:
        ResponseBase dengan status penghapusan
    """
    # Check if lokasi exists and belongs to pedagang
    lokasi = db_manager.get_lokasi_by_id(db, lokasi_id)
    if not lokasi or lokasi.pedagang_id != pedagang.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lokasi tidak ditemukan"
        )
    
    deleted = db_manager.delete_lokasi(db, lokasi_id)
    
    if deleted:
        return ResponseBase(
            success=True,
            message="Lokasi berhasil dihapus"
        )
    else:
        return ResponseBase(
            success=False,
            message="Gagal menghapus lokasi"
        )


# ==================== Transaksi Endpoints (Sistem Baru) ====================

@app.post("/transaksi", response_model=TransaksiCreateResponse, tags=["Transaksi"])
async def catat_transaksi(
    request: TransaksiCreate,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    **Endpoint utama** untuk mencatat transaksi per konsumen.

    Backend secara otomatis:
    1. Mengecek apakah ada kunjungan aktif
    2. Jika tidak ada → membuat kunjungan baru di lokasi ini
    3. Jika ada tapi lokasi BERBEDA → menutup kunjungan lama, membuat baru
    4. Jika ada dan lokasi SAMA → menggunakan kunjungan yang ada
    5. Menyimpan transaksi ke tabel penjualan

    **Durasi mangkal dihitung otomatis** — tidak perlu dikirim dari frontend.
    """
    # Verifikasi lokasi milik pedagang
    lokasi = db_manager.get_lokasi_by_id(db, request.lokasi_id)
    if not lokasi or lokasi.pedagang_id != pedagang.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lokasi tidak ditemukan atau bukan milik pedagang ini"
        )

    # Validasi: harus ada sesi aktif sebelum bisa input transaksi
    sesi_aktif = db_manager.get_active_sesi(db, pedagang.id)
    if not sesi_aktif:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Silakan Start Berjualan terlebih dahulu sebelum menginput transaksi"
        )

    # Proses transaksi — orkestrasi kunjungan otomatis
    result = kunjungan_service.process_transaksi(
        db=db,
        pedagang_id=pedagang.id,
        lokasi_id=request.lokasi_id,
        jumlah_terjual=request.jumlah_terjual,
        kondisi_cuaca=sesi_aktif.kondisi_cuaca,
        hari_kuliah=sesi_aktif.hari_kuliah
    )

    transaksi = result["transaksi"]
    kunjungan = result["kunjungan"]

    # Buat pesan yang informatif
    if result["lokasi_berganti"]:
        pesan = f"Pindah ke {lokasi.nama} — kunjungan baru dimulai. Transaksi dicatat."
    elif result["kunjungan_baru"]:
        pesan = f"Kunjungan baru di {lokasi.nama} dimulai. Transaksi dicatat."
    else:
        pesan = f"Transaksi dicatat di {lokasi.nama} (kunjungan berlangsung)."

    return TransaksiCreateResponse(
        success=True,
        message=pesan,
        transaksi=TransaksiResponse(
            id=transaksi.id,
            kunjungan_id=transaksi.kunjungan_id,
            lokasi_id=transaksi.lokasi_id,
            jumlah_terjual=transaksi.jumlah_terjual,
            waktu_transaksi=transaksi.waktu_transaksi
        ),
        kunjungan=KunjunganResponse(
            id=kunjungan.id,
            pedagang_id=kunjungan.pedagang_id,
            lokasi_id=kunjungan.lokasi_id,
            waktu_mulai=kunjungan.waktu_mulai,
            waktu_selesai=kunjungan.waktu_selesai,
            durasi_mangkal=kunjungan.durasi_mangkal,
            kondisi_cuaca=kunjungan.kondisi_cuaca,
            hari_kuliah=kunjungan.hari_kuliah,
            sedang_aktif=(kunjungan.waktu_selesai is None)
        ),
        kunjungan_baru=result["kunjungan_baru"],
        lokasi_berganti=result["lokasi_berganti"]
    )


# ==================== Kunjungan Endpoints ====================

# =========================================================================================
# KUNJUNGAN & TRANSAKSI
# =========================================================================================

@app.post("/kunjungan/pindah", response_model=KunjunganDetailResponse, tags=["Kunjungan"])
async def pindah_lokasi(
    request: PindahLokasiRequest,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    **Pindah ke lokasi pangkalan baru secara eksplisit tanpa harus mencatat transaksi terlebih dahulu.**
    Jika ada kunjungan/mangkal aktif di lokasi sebelumnya, kunjungan lama akan ditutup (durasi dihitung).
    Lalu membuka sesi mangkal baru di lokasi tujuan.
    """
    from app.services import kunjungan_service
    from app.models import SesiPenjualan, Kunjungan
    from sqlalchemy.orm import joinedload
    
    # Cari sesi aktif
    active_sesi = db.query(SesiPenjualan).filter(
        SesiPenjualan.pedagang_id == pedagang.id,
        SesiPenjualan.waktu_selesai == None
    ).first()
    sesi_id = active_sesi.id if active_sesi else None

    # Cek kunjungan aktif
    active = kunjungan_service.get_active_kunjungan(db, pedagang.id)
    
    if active and active.lokasi_id == request.lokasi_id:
        kunjungan = active
    else:
        if active:
            kunjungan_service.close_kunjungan(db, active)
            
        kunjungan = kunjungan_service.create_kunjungan(
            db, pedagang.id, request.lokasi_id, 
            request.kondisi_cuaca, request.hari_kuliah, 
            sesi_id=sesi_id
        )
        
    kunjungan_with_lokasi = db.query(Kunjungan).options(
        joinedload(Kunjungan.lokasi)
    ).filter(Kunjungan.id == kunjungan.id).first()
    
    if not kunjungan_with_lokasi:
        raise HTTPException(status_code=500, detail="Gagal mengambil data kunjungan")
        
    # Tambahkan dummy total_pendapatan jika diperlukan schema
    kunjungan_with_lokasi.total_pendapatan = 0
    return kunjungan_with_lokasi

@app.get("/kunjungan", response_model=KunjunganListResponse, tags=["Kunjungan"])
async def get_kunjungan(
    limit: int = 20,
    offset: int = 0,
    lokasi_id: int | None = None,
    hari_kuliah: int | None = None,
    hanya_selesai: bool = False,
    hanya_aktif: bool = False,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Mendapatkan riwayat sesi kunjungan/mangkal pedagang.

    - **hanya_selesai=true** → hanya tampilkan kunjungan yang sudah ditutup
    - **hanya_aktif=true** → hanya tampilkan kunjungan yang masih berlangsung
    """
    kunjungan_list = db_manager.get_kunjungan_list(
        db=db,
        pedagang_id=pedagang.id,
        limit=limit,
        offset=offset,
        lokasi_id=lokasi_id,
        hari_kuliah=hari_kuliah,
        hanya_selesai=hanya_selesai,
        hanya_aktif=hanya_aktif
    )

    return KunjunganListResponse(
        success=True,
        message=f"Ditemukan {len(kunjungan_list)} kunjungan",
        data=[
            KunjunganResponse(
                id=k.id,
                pedagang_id=k.pedagang_id,
                lokasi_id=k.lokasi_id,
                waktu_mulai=k.waktu_mulai,
                waktu_selesai=k.waktu_selesai,
                durasi_mangkal=k.durasi_mangkal,
                kondisi_cuaca=k.kondisi_cuaca,
                hari_kuliah=k.hari_kuliah,
                sedang_aktif=(k.waktu_selesai is None),
                total_pendapatan=sum(t.jumlah_terjual for t in k.transaksi)
            ) for k in kunjungan_list
        ]
    )


@app.get("/kunjungan/aktif", response_model=KunjunganAktifResponse, tags=["Kunjungan"])
async def get_kunjungan_aktif(
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Mendapatkan kunjungan yang sedang aktif (jika ada).
    Returns null jika tidak ada kunjungan yang berlangsung.
    """
    active = db_manager.get_active_kunjungan(db, pedagang.id)

    if not active:
        return KunjunganAktifResponse(
            success=True,
            message="Tidak ada kunjungan yang sedang aktif",
            data=None
        )

    # Ambil detail lokasi
    lokasi = db_manager.get_lokasi_by_id(db, active.lokasi_id)
    lokasi_data = None
    if lokasi:
        lokasi_data = LokasiResponse(
            id=lokasi.id,
            pedagang_id=lokasi.pedagang_id,
            nama=lokasi.nama,
            latitude=lokasi.latitude,
            longitude=lokasi.longitude,
            created_at=lokasi.created_at
        )

    return KunjunganAktifResponse(
        success=True,
        message=f"Kunjungan aktif di lokasi ID {active.lokasi_id}",
        data=KunjunganDetailResponse(
            id=active.id,
            pedagang_id=active.pedagang_id,
            lokasi_id=active.lokasi_id,
            waktu_mulai=active.waktu_mulai,
            waktu_selesai=active.waktu_selesai,
            durasi_mangkal=active.durasi_mangkal,
            kondisi_cuaca=active.kondisi_cuaca,
            hari_kuliah=active.hari_kuliah,
            sedang_aktif=True,
            lokasi=lokasi_data
        )
    )


@app.post("/kunjungan/{kunjungan_id}/selesai", response_model=KunjunganAktifResponse, tags=["Kunjungan"])
async def selesai_kunjungan(
    kunjungan_id: int,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Menutup kunjungan secara manual (pedagang selesai mangkal).
    Durasi dihitung otomatis dari waktu_mulai sampai sekarang.
    """
    kunjungan = db_manager.get_kunjungan_by_id(db, kunjungan_id)

    if not kunjungan or kunjungan.pedagang_id != pedagang.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kunjungan tidak ditemukan"
        )

    if kunjungan.waktu_selesai is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kunjungan sudah ditutup sebelumnya"
        )

    kunjungan = db_manager.close_kunjungan(db, kunjungan)

    return KunjunganAktifResponse(
        success=True,
        message=f"Kunjungan selesai. Durasi mangkal: {kunjungan.durasi_mangkal} menit",
        data=KunjunganDetailResponse(
            id=kunjungan.id,
            pedagang_id=kunjungan.pedagang_id,
            lokasi_id=kunjungan.lokasi_id,
            waktu_mulai=kunjungan.waktu_mulai,
            waktu_selesai=kunjungan.waktu_selesai,
            durasi_mangkal=kunjungan.durasi_mangkal,
            kondisi_cuaca=kunjungan.kondisi_cuaca,
            hari_kuliah=kunjungan.hari_kuliah,
            sedang_aktif=False
        )
    )


# ==================== Sesi Penjualan Endpoints (Start/Stop) ====================

@app.post("/sesi/start", response_model=SesiAktifResponse, tags=["Sesi Penjualan"])
async def start_sesi(
    request: SesiPenjualanStartRequest,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    **Memulai sesi berjualan hari ini (Start Berjualan).**

    Pedagang harus memanggil endpoint ini sebelum bisa menginput transaksi.
    Jika sudah ada sesi aktif, akan mengembalikan error.

    Args:
        request: kondisi_cuaca, hari_kuliah, stok_awal (opsional)
    """
    # Cek apakah sudah ada sesi aktif
    existing = db_manager.get_active_sesi(db, pedagang.id)
    if existing:
        from app.models import Kunjungan, Penjualan
        from sqlalchemy import func
        kunjungan_ids = db.query(Kunjungan.id).filter(Kunjungan.sesi_id == existing.id).all()
        k_ids = [k[0] for k in kunjungan_ids]
        
        tot_pendapatan = 0
        tot_transaksi = 0
        if k_ids:
            hasil = db.query(
                func.sum(Penjualan.jumlah_terjual).label('tot_pend'),
                func.count(Penjualan.id).label('tot_trans')
            ).filter(Penjualan.kunjungan_id.in_(k_ids)).first()
            if hasil:
                tot_pendapatan = int(hasil.tot_pend or 0)
                tot_transaksi = int(hasil.tot_trans or 0)

        return SesiAktifResponse(
            success=False,
            message="Sudah ada sesi berjualan yang aktif. Stop sesi sebelumnya terlebih dahulu.",
            data=SesiPenjualanResponse(
                id=existing.id,
                pedagang_id=existing.pedagang_id,
                waktu_mulai=existing.waktu_mulai,
                waktu_selesai=existing.waktu_selesai,
                stok_awal=existing.stok_awal,
                stok_sisa=existing.stok_sisa,
                total_pendapatan=tot_pendapatan,
                total_transaksi=tot_transaksi,
                total_lokasi_dikunjungi=existing.total_lokasi_dikunjungi,
                durasi_total=existing.durasi_total,
                kondisi_cuaca=existing.kondisi_cuaca,
                hari_kuliah=existing.hari_kuliah,
                sedang_aktif=True
            )
        )

    # Ensure Basecamp exists
    lokasi_list = db_manager.get_lokasi(db, pedagang.id)
    basecamp = next((l for l in lokasi_list if l.nama.lower() == "basecamp"), None)
    if not basecamp:
        basecamp = db_manager.create_lokasi(db, pedagang.id, "Basecamp", -6.200000, 106.816666)

    # Auto cuaca dan hari
    kondisi_cuaca = request.kondisi_cuaca or get_cuaca_otomatis(basecamp.latitude, basecamp.longitude)
    hari_kuliah = request.hari_kuliah if request.hari_kuliah is not None else get_hari_otomatis()

    # Buat sesi baru
    sesi = db_manager.create_sesi_penjualan(
        db=db,
        pedagang_id=pedagang.id,
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
        stok_awal=request.stok_awal
    )

    # Jalankan optimasi rute otomatis
    controller = SistemController(db, db_manager, pedagang.id)
    result = controller.jalankan_q_learning(
        max_episodes=100,
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
        start_lokasi_id=basecamp.id
    )

    # Convert ke format OptimasiResponse
    rute_optimal = []
    if result.get("rute_optimal"):
        for loc in result["rute_optimal"]:
            rute_optimal.append(RuteLokasiResponse(
                urutan=loc["urutan"],
                lokasi_id=loc["lokasi_id"],
                nama=loc["nama"],
                latitude=loc["latitude"],
                longitude=loc["longitude"],
                jarak_dari_sebelumnya=loc.get("jarak_dari_sebelumnya", 0.0)
            ))

    durasi_rekomendasi = []
    if result.get("durasi_per_lokasi"):
        for d in result["durasi_per_lokasi"]:
            durasi_rekomendasi.append(DurasiLokasiResponse(
                lokasi_id=d["lokasi_id"],
                nama=d["nama"],
                durasi_menit=d["durasi_menit"],
                reward=d.get("reward", 0.0)
            ))

    reward_info = proses_reward(
        lokasi_stats=result.get("lokasi_stats", {}),
        route_lokasi_ids=result.get("route_lokasi_ids", []),
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
    )

    optimasi_res = OptimasiResponse(
        success=result.get("success", False),
        message=result.get("message", ""),
        rekomendasi_rute=rute_optimal,
        total_jarak_km=result.get("total_jarak", 0.0),
        perkiraan_penghasilan=int(reward_info["perkiraan_penghasilan"]),
        kategori=reward_info["kategori"],
        penjelasan=reward_info["penjelasan"],
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
        durasi_rekomendasi=durasi_rekomendasi,
        total_reward_lokasi=result.get("total_reward", 0.0),
        rute_optimal=rute_optimal,
        total_reward=result.get("total_reward", 0.0),
        total_jarak=result.get("total_jarak", 0.0),
        rekomendasi=result.get("rekomendasi", ""),
        episode_rewards=result.get("episode_rewards", [])
    )

    return SesiAktifResponse(
        success=True,
        message="Sesi berjualan dimulai! Selamat berjualan hari ini.",
        data=SesiPenjualanResponse(
            id=sesi.id,
            pedagang_id=sesi.pedagang_id,
            waktu_mulai=sesi.waktu_mulai,
            waktu_selesai=sesi.waktu_selesai,
            stok_awal=sesi.stok_awal,
            stok_sisa=sesi.stok_sisa,
            total_pendapatan=sesi.total_pendapatan,
            total_transaksi=sesi.total_transaksi,
            total_lokasi_dikunjungi=sesi.total_lokasi_dikunjungi,
            durasi_total=sesi.durasi_total,
            kondisi_cuaca=sesi.kondisi_cuaca,
            hari_kuliah=sesi.hari_kuliah,
            sedang_aktif=True
        ),
        optimasi=optimasi_res
    )


@app.post("/sesi/stop", response_model=SesiRingkasanResponse, tags=["Sesi Penjualan"])
async def stop_sesi(
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    **Mengakhiri sesi berjualan (Stop / Pulang).**

    - Menutup semua kunjungan aktif
    - Menghitung ringkasan: total pendapatan, transaksi, durasi
    - Mengembalikan ringkasan sesi
    """
    sesi = db_manager.get_active_sesi(db, pedagang.id)
    if not sesi:
        return SesiRingkasanResponse(
            success=False,
            message="Tidak ada sesi berjualan yang aktif.",
            data=None,
            kunjungan_list=[]
        )

    # Tutup sesi — ini juga menutup kunjungan aktif dan menghitung ringkasan
    sesi = db_manager.close_sesi_penjualan(db, sesi)

    # Ambil daftar kunjungan dalam sesi ini
    kunjungan_list = db_manager.get_kunjungan_by_sesi(db, sesi.id)

    return SesiRingkasanResponse(
        success=True,
        message=f"Sesi berjualan selesai! Total pendapatan: Rp {sesi.total_pendapatan:,}",
        data=SesiPenjualanResponse(
            id=sesi.id,
            pedagang_id=sesi.pedagang_id,
            waktu_mulai=sesi.waktu_mulai,
            waktu_selesai=sesi.waktu_selesai,
            stok_awal=sesi.stok_awal,
            stok_sisa=sesi.stok_sisa,
            total_pendapatan=sesi.total_pendapatan,
            total_transaksi=sesi.total_transaksi,
            total_lokasi_dikunjungi=sesi.total_lokasi_dikunjungi,
            durasi_total=sesi.durasi_total,
            kondisi_cuaca=sesi.kondisi_cuaca,
            hari_kuliah=sesi.hari_kuliah,
            sedang_aktif=False
        ),
        kunjungan_list=[
            KunjunganResponse(
                id=k.id,
                pedagang_id=k.pedagang_id,
                lokasi_id=k.lokasi_id,
                waktu_mulai=k.waktu_mulai,
                waktu_selesai=k.waktu_selesai,
                durasi_mangkal=k.durasi_mangkal,
                kondisi_cuaca=k.kondisi_cuaca,
                hari_kuliah=k.hari_kuliah,
                sedang_aktif=False,
                total_pendapatan=sum(t.jumlah_terjual for t in k.transaksi)
            ) for k in kunjungan_list
        ]
    )


@app.get("/sesi/aktif", response_model=SesiAktifResponse, tags=["Sesi Penjualan"])
async def get_sesi_aktif(
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    **Mengecek apakah ada sesi berjualan yang sedang aktif.**

    Returns null jika tidak ada sesi aktif (pedagang belum Start).
    """
    sesi = db_manager.get_active_sesi(db, pedagang.id)

    if not sesi:
        return SesiAktifResponse(
            success=True,
            message="Tidak ada sesi berjualan yang aktif.",
            data=None
        )

    from app.models import Kunjungan, Penjualan
    from sqlalchemy import func
    kunjungan_ids = db.query(Kunjungan.id).filter(Kunjungan.sesi_id == sesi.id).all()
    k_ids = [k[0] for k in kunjungan_ids]
    
    tot_pendapatan = 0
    tot_transaksi = 0
    if k_ids:
        hasil = db.query(
            func.sum(Penjualan.jumlah_terjual).label('tot_pend'),
            func.count(Penjualan.id).label('tot_trans')
        ).filter(Penjualan.kunjungan_id.in_(k_ids)).first()
        if hasil:
            tot_pendapatan = int(hasil.tot_pend or 0)
            tot_transaksi = int(hasil.tot_trans or 0)

    return SesiAktifResponse(
        success=True,
        message=f"Sesi berjualan aktif sejak {sesi.waktu_mulai}",
        data=SesiPenjualanResponse(
            id=sesi.id,
            pedagang_id=sesi.pedagang_id,
            waktu_mulai=sesi.waktu_mulai,
            waktu_selesai=sesi.waktu_selesai,
            stok_awal=sesi.stok_awal,
            stok_sisa=sesi.stok_sisa,
            total_pendapatan=tot_pendapatan,
            total_transaksi=tot_transaksi,
            total_lokasi_dikunjungi=sesi.total_lokasi_dikunjungi,
            durasi_total=sesi.durasi_total,
            kondisi_cuaca=sesi.kondisi_cuaca,
            hari_kuliah=sesi.hari_kuliah,
            sedang_aktif=True
        )
    )


# ==================== Penjualan Endpoints (Legacy + Backward Compat) ====================

@app.post("/penjualan", response_model=PenjualanCreateResponse, tags=["Penjualan"])
async def create_penjualan(
    request: PenjualanCreate,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Mencatat hasil penjualan harian sebagai umpan balik reward.
    
    Args:
        request: PenjualanCreate dengan lokasi_id, jumlah_terjual, durasi_mangkal
        
    Returns:
        PenjualanCreateResponse dengan data penjualan yang dibuat
    """
    # Verify lokasi belongs to pedagang
    lokasi = db_manager.get_lokasi_by_id(db, request.lokasi_id)
    if not lokasi or lokasi.pedagang_id != pedagang.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lokasi tidak ditemukan"
        )
        
    kondisi_cuaca = request.kondisi_cuaca or get_cuaca_otomatis(lokasi.latitude, lokasi.longitude)
    hari_kuliah = request.hari_kuliah if request.hari_kuliah is not None else get_hari_otomatis()
    
    penjualan = db_manager.create_penjualan(
        db,
        pedagang_id=pedagang.id,
        lokasi_id=request.lokasi_id,
        jumlah_terjual=request.jumlah_terjual,
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah
    )
    
    return PenjualanCreateResponse(
        success=True,
        message=f"Penjualan di {lokasi.nama} berhasil dicatat",
        data=PenjualanResponse(
            id=penjualan.id,
            pedagang_id=penjualan.pedagang_id,
            lokasi_id=penjualan.lokasi_id,
            waktu_kunjungan=penjualan.waktu_kunjungan,
            jumlah_terjual=penjualan.jumlah_terjual,
            durasi_mangkal=penjualan.durasi_mangkal,
            kondisi_cuaca=penjualan.kondisi_cuaca,
            hari_kuliah=penjualan.hari_kuliah
        )
    )


@app.get("/penjualan", response_model=PenjualanListResponse, tags=["Penjualan"])
async def get_penjualan(
    limit: int = 20,
    offset: int = 0,
    lokasi_id: int | None = None,
    hari_kuliah: int | None = None,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    
    penjualan_list = db_manager.get_penjualan(
        db = db, 
        pedagang_id = pedagang.id,
        limit = limit,
        offset = offset,
        lokasi_id = lokasi_id,
        hari_kuliah = hari_kuliah
        )
    
    return PenjualanListResponse(
        success=True,
        message=f"Ditemukan {len(penjualan_list)} data penjualan",
        data=[PenjualanResponse(
            id=p.id,
            pedagang_id=p.pedagang_id,
            lokasi_id=p.lokasi_id,
            waktu_kunjungan=p.waktu_kunjungan,
            jumlah_terjual=p.jumlah_terjual,
            durasi_mangkal=p.durasi_mangkal,
            kondisi_cuaca=p.kondisi_cuaca,
            hari_kuliah=p.hari_kuliah
        ) for p in penjualan_list]
    )


@app.get("/penjualan/performa", response_model=PerformaPenjualanResponse, tags=["Penjualan"])
async def get_performa_penjualan(
    days: int = 7,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Performa penjualan per hari — total nominal belanja semua konsumen.

    `jumlah_terjual` menyimpan nominal belanja per konsumen (rupiah).
    Backend menjumlahkan semua transaksi konsumen per hari menjadi `total_pendapatan`.
    Tidak ada konversi harga per porsi — nilai langsung dari input frontend.

    Args:
        days: Jumlah hari terakhir yang ingin diambil (default 7)
    """
    results = db_manager.get_performa_penjualan(db, pedagang.id, days)
    
    performa_data = [
        PerformaHarian(
            tanggal=str(r.tanggal),
            total_pendapatan=int(r.total_pendapatan or 0),
            total_transaksi=int(r.total_transaksi or 0),
        )
        for r in results
    ]
        
    return PerformaPenjualanResponse(
        success=True,
        message=f"Berhasil mengambil data performa penjualan {days} hari terakhir",
        data=performa_data
    )


# ==================== Statistik Kunjungan Endpoints ====================

@app.get("/kunjungan/statistik", tags=["Kunjungan"])
async def get_statistik_kunjungan(
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Statistik kunjungan per lokasi untuk seluruh data historis.
    Berguna untuk frontend menampilkan ringkasan data dari import Excel.

    Returns:
        List statistik per lokasi: nama, total kunjungan, total durasi,
        rata-rata durasi, total pendapatan, rata-rata pendapatan.
    """
    from sqlalchemy import func
    from app.models import Kunjungan, Penjualan, Lokasi as LokasiModel

    hasil = db.query(
        LokasiModel.id.label("lokasi_id"),
        LokasiModel.nama.label("nama"),
        LokasiModel.latitude,
        LokasiModel.longitude,
        func.count(Kunjungan.id).label("total_kunjungan"),
        func.sum(Kunjungan.durasi_mangkal).label("total_durasi_menit"),
        func.avg(Kunjungan.durasi_mangkal).label("rata_durasi_menit"),
        func.sum(Penjualan.jumlah_terjual).label("total_pendapatan"),
        func.avg(Penjualan.jumlah_terjual).label("rata_pendapatan")
    ).join(
        Kunjungan, Kunjungan.lokasi_id == LokasiModel.id
    ).outerjoin(
        Penjualan, Penjualan.kunjungan_id == Kunjungan.id
    ).filter(
        LokasiModel.pedagang_id == pedagang.id
    ).group_by(
        LokasiModel.id, LokasiModel.nama, LokasiModel.latitude, LokasiModel.longitude
    ).order_by(
        func.sum(Penjualan.jumlah_terjual).desc()
    ).all()

    data = [
        {
            "lokasi_id": r.lokasi_id,
            "nama": r.nama,
            "latitude": r.latitude,
            "longitude": r.longitude,
            "total_kunjungan": r.total_kunjungan or 0,
            "total_durasi_menit": int(r.total_durasi_menit or 0),
            "rata_durasi_menit": round(float(r.rata_durasi_menit or 0), 1),
            "total_pendapatan": int(r.total_pendapatan or 0),
            "rata_pendapatan": round(float(r.rata_pendapatan or 0), 0),
        }
        for r in hasil
    ]

    return {
        "success": True,
        "message": f"Statistik dari {len(data)} lokasi",
        "data": data
    }


@app.get("/kunjungan/histori-harian", tags=["Kunjungan"])
async def get_histori_harian(
    days: int = 30,
    lokasi_id: int | None = None,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Riwayat kunjungan harian dengan total pendapatan per hari.
    Berguna untuk grafik tren di frontend.

    Args:
        days: Ambil N hari terakhir (default 30). Gunakan 0 untuk semua data.
        lokasi_id: Filter per lokasi (opsional)
    """
    from sqlalchemy import func
    from app.models import Kunjungan, Penjualan

    query = db.query(
        func.date(Kunjungan.waktu_mulai).label("tanggal"),
        func.count(Kunjungan.id).label("total_kunjungan"),
        func.sum(Kunjungan.durasi_mangkal).label("total_durasi"),
        func.sum(Penjualan.jumlah_terjual).label("total_pendapatan"),
        Kunjungan.kondisi_cuaca,
        Kunjungan.hari_kuliah
    ).outerjoin(
        Penjualan, Penjualan.kunjungan_id == Kunjungan.id
    ).filter(
        Kunjungan.pedagang_id == pedagang.id
    )

    if lokasi_id is not None:
        query = query.filter(Kunjungan.lokasi_id == lokasi_id)

    query = query.group_by(
        func.date(Kunjungan.waktu_mulai),
        Kunjungan.kondisi_cuaca,
        Kunjungan.hari_kuliah
    ).order_by(
        func.date(Kunjungan.waktu_mulai).desc()
    )

    if days > 0:
        query = query.limit(days)

    hasil = query.all()
    hasil.reverse()  # chronological order

    data = [
        {
            "tanggal": str(r.tanggal),
            "total_kunjungan": r.total_kunjungan or 0,
            "total_durasi_menit": int(r.total_durasi or 0),
            "total_pendapatan": int(r.total_pendapatan or 0),
            "kondisi_cuaca": r.kondisi_cuaca,
            "hari_kuliah": r.hari_kuliah
        }
        for r in hasil
    ]

    return {
        "success": True,
        "message": f"Data historis {len(data)} hari",
        "data": data
    }


# ==================== Optimasi Endpoints ====================

@app.get("/optimasi", response_model=OptimasiResponse, tags=["Optimasi"])
async def get_optimasi(
    kondisi_cuaca: str | None = None,
    hari_kuliah: int | None = None,
    max_episodes: int = 100,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Menjalankan proses pembelajaran Q-Learning dan mengembalikan rute/prioritas singgah optimal.
    
    Args:
        kondisi_cuaca: Kondisi cuaca (opsional, otomatis jika None)
        hari_kuliah: 1 = hari kuliah, 0 = akhir pekan (opsional, otomatis jika None)
        max_episodes: Jumlah episode training (default 100)
        
    Returns:
        OptimasiResponse dengan rute optimal, perkiraan penghasilan, dan kategori
    """
    lokasi_list = db_manager.get_lokasi(db, pedagang.id)
    basecamp = next((l for l in lokasi_list if l.nama.lower() == "basecamp"), None)
    lat, lon = (-6.200000, 106.816666)
    if basecamp:
        lat, lon = basecamp.latitude, basecamp.longitude
        
    if kondisi_cuaca is None:
        kondisi_cuaca = get_cuaca_otomatis(lat, lon)
    if hari_kuliah is None:
        hari_kuliah = get_hari_otomatis()

    # Initialize sistem controller
    controller = SistemController(db, db_manager, pedagang.id)
    
    # Run Q-Learning optimization
    result = controller.jalankan_q_learning(
        max_episodes=max_episodes,
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
        start_lokasi_id=basecamp.id if basecamp else None
    )
    
    # Convert to response schema
    rute_optimal = []
    if result.get("rute_optimal"):
        for loc in result["rute_optimal"]:
            rute_optimal.append(RuteLokasiResponse(
                urutan=loc["urutan"],
                lokasi_id=loc["lokasi_id"],
                nama=loc["nama"],
                latitude=loc["latitude"],
                longitude=loc["longitude"],
                jarak_dari_sebelumnya=loc.get("jarak_dari_sebelumnya", 0.0)
            ))
    
    # Convert durasi per lokasi
    durasi_rekomendasi = []
    if result.get("durasi_per_lokasi"):
        for d in result["durasi_per_lokasi"]:
            durasi_rekomendasi.append(DurasiLokasiResponse(
                lokasi_id=d["lokasi_id"],
                nama=d["nama"],
                durasi_menit=d["durasi_menit"],
                reward=d.get("reward", 0.0)
            ))
    
    # Hitung perkiraan penghasilan, kategori, dan penjelasan
    reward_info = proses_reward(
        lokasi_stats=result.get("lokasi_stats", {}),
        route_lokasi_ids=result.get("route_lokasi_ids", []),
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
    )
    
    return OptimasiResponse(
        success=result.get("success", False),
        message=result.get("message", ""),
        # Field baru (frontend-friendly)
        rekomendasi_rute=rute_optimal,
        total_jarak_km=result.get("total_jarak", 0.0),
        perkiraan_penghasilan=reward_info["perkiraan_penghasilan"],
        kategori=reward_info["kategori"],
        penjelasan=reward_info["penjelasan"],
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
        # Field baru: durasi rekomendasi per lokasi
        durasi_rekomendasi=durasi_rekomendasi,
        total_reward_lokasi=result.get("total_reward_lokasi", 0.0),
        # Field lama (backward compat)
        rute_optimal=rute_optimal,
        total_reward=result.get("total_reward", 0.0),
        total_jarak=result.get("total_jarak", 0.0),
        rekomendasi=result.get("rekomendasi", ""),
        episode_rewards=result.get("episode_rewards", [])
    )


@app.post("/optimasi", response_model=OptimasiResponse, tags=["Optimasi"])
async def run_optimasi(
    request: OptimasiRequest,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Menjalankan proses pembelajaran Q-Learning dengan parameter kustom.
    
    Args:
        request: OptimasiRequest dengan parameter training
        
    Returns:
        OptimasiResponse dengan rute optimal, perkiraan penghasilan, dan kategori
    """
    kondisi_cuaca = request.kondisi_cuaca or "cerah"
    hari_kuliah = request.hari_kuliah if request.hari_kuliah is not None else 1
    
    # Initialize sistem controller
    controller = SistemController(db, db_manager, pedagang.id)
    
    # Run Q-Learning optimization
    result = controller.jalankan_q_learning(
        max_episodes=request.max_episodes or 100,
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah
    )
    
    # Convert to response schema
    rute_optimal = []
    if result.get("rute_optimal"):
        for loc in result["rute_optimal"]:
            rute_optimal.append(RuteLokasiResponse(
                urutan=loc["urutan"],
                lokasi_id=loc["lokasi_id"],
                nama=loc["nama"],
                latitude=loc["latitude"],
                longitude=loc["longitude"],
                jarak_dari_sebelumnya=loc.get("jarak_dari_sebelumnya", 0.0)
            ))
    
    # Convert durasi per lokasi
    durasi_rekomendasi = []
    if result.get("durasi_per_lokasi"):
        for d in result["durasi_per_lokasi"]:
            durasi_rekomendasi.append(DurasiLokasiResponse(
                lokasi_id=d["lokasi_id"],
                nama=d["nama"],
                durasi_menit=d["durasi_menit"],
                reward=d.get("reward", 0.0)
            ))
    
    # Hitung perkiraan penghasilan, kategori, dan penjelasan
    reward_info = proses_reward(
        lokasi_stats=result.get("lokasi_stats", {}),
        route_lokasi_ids=result.get("route_lokasi_ids", []),
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
    )
    
    return OptimasiResponse(
        success=result.get("success", False),
        message=result.get("message", ""),
        # Field baru (frontend-friendly)
        rekomendasi_rute=rute_optimal,
        total_jarak_km=result.get("total_jarak", 0.0),
        perkiraan_penghasilan=reward_info["perkiraan_penghasilan"],
        kategori=reward_info["kategori"],
        penjelasan=reward_info["penjelasan"],
        kondisi_cuaca=kondisi_cuaca,
        hari_kuliah=hari_kuliah,
        # Field baru: durasi rekomendasi per lokasi
        durasi_rekomendasi=durasi_rekomendasi,
        total_reward_lokasi=result.get("total_reward_lokasi", 0.0),
        # Field lama (backward compat)
        rute_optimal=rute_optimal,
        total_reward=result.get("total_reward", 0.0),
        total_jarak=result.get("total_jarak", 0.0),
        rekomendasi=result.get("rekomendasi", ""),
        episode_rewards=result.get("episode_rewards", [])
    )


@app.post("/optimasi/selanjutnya", response_model=RekomendasiSelanjutnyaResponse, tags=["Optimasi"])
async def get_rekomendasi_selanjutnya(
    request: RekomendasiSelanjutnyaRequest,
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Meminta rekomendasi AI secara real-time (STAY vs MOVE) di tengah sesi.
    """
    sesi_aktif = db_manager.get_active_sesi(db, pedagang.id)
    if not sesi_aktif:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tidak ada sesi berjualan aktif. Start berjualan terlebih dahulu."
        )
        
    controller = SistemController(db, db_manager, pedagang.id)
    result = controller.dapatkan_rekomendasi_selanjutnya(
        lokasi_saat_ini_id=request.lokasi_saat_ini_id,
        sisa_waktu_menit=request.sisa_waktu_menit or 120,
        kondisi_cuaca=sesi_aktif.kondisi_cuaca,
        hari_kuliah=sesi_aktif.hari_kuliah
    )
    
    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("message", "Gagal mendapatkan rekomendasi.")
        )
        
    return RekomendasiSelanjutnyaResponse(
        success=True,
        message=result.get("message"),
        keputusan=result.get("keputusan"),
        lokasi_tujuan_id=result.get("lokasi_tujuan_id"),
        nama_lokasi_tujuan=result.get("nama_lokasi_tujuan"),
        rekomendasi_durasi_menit=result.get("rekomendasi_durasi_menit"),
        alasan=result.get("alasan")
    )


# ==================== Episode Endpoints ====================

@app.get("/episodes", response_model=EpisodeListResponse, tags=["Episode"])
async def get_episodes(
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Mendapatkan riwayat episode training.
    
    Returns:
        EpisodeListResponse dengan daftar episode
    """
    episodes = db_manager.get_episodes(db, pedagang.id)
    
    return EpisodeListResponse(
        success=True,
        message=f"Ditemukan {len(episodes)} episode",
        data=[EpisodeResponse(
            id=e.id,
            pedagang_id=e.pedagang_id,
            total_reward=e.total_reward,
            total_waktu=e.total_waktu,
            created_at=e.created_at
        ) for e in episodes]
    )


# ==================== Rute Optimal Endpoints ====================

@app.get("/rute-optimal", response_model=RuteOptimalDetailResponse, tags=["Rute Optimal"])
async def get_rute_optimal(
    pedagang: Pedagang = Depends(get_current_pedagang),
    db: Session = Depends(db_manager.get_db)
):
    """
    Mendapatkan rute optimal yang tersimpan.
    
    Returns:
        RuteOptimalDetailResponse dengan rute dan detail lokasi
    """
    rute = db_manager.get_rute_optimal(db, pedagang.id)
    
    if not rute:
        return RuteOptimalDetailResponse(
            success=False,
            message="Belum ada rute optimal. Jalankan optimasi terlebih dahulu.",
            data=None,
            lokasi_details=[]
        )
    
    # Get lokasi details
    lokasi_list = db_manager.get_lokasi(db, pedagang.id)
    lokasi_details = []
    
    for lokasi_id in rute.urutan_lokasi:
        lokasi = next((l for l in lokasi_list if l.id == lokasi_id), None)
        if lokasi:
            lokasi_details.append(LokasiResponse(
                id=lokasi.id,
                pedagang_id=lokasi.pedagang_id,
                nama=lokasi.nama,
                latitude=lokasi.latitude,
                longitude=lokasi.longitude,
                created_at=lokasi.created_at
            ))
    
    return RuteOptimalDetailResponse(
        success=True,
        message="Rute optimal ditemukan",
        data=RuteOptimalResponse(
            id=rute.id,
            pedagang_id=rute.pedagang_id,
            urutan_lokasi=rute.urutan_lokasi,
            total_jarak=rute.total_jarak,
            total_reward=rute.total_reward,
            rekomendasi=rute.rekomendasi,
            created_at=rute.created_at
        ),
        lokasi_details=lokasi_details
    )


# ==================== Health Check ====================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

# @app.get("/qlearning")
# def run_qlearning(db: Session = Depends(get_db)):
#     try:
#         data_db = db.query(Penjualan).all()

#         data = []
#         for d in data_db:
#             data.append({
#                 "id_lokasi": d.id_lokasi,
#                 "jumlah_terjual": d.jumlah_terjual,
#                 "kondisi_cuaca": d.kondisi_cuaca,
#                 "hari_kuliah": d.hari_kuliah
#             })

#         hasil = SistemController.jalankan_q_learning(data)

#         return hasil

#     except Exception as e:
#         return {
#             "error": str(e)
#         }

# ==================== Run Application ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
