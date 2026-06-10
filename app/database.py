"""
Database Manager for MySQL connection and operations.
Implements the DatabaseManager class from the Class Diagram.
"""

from typing import List, Optional
from datetime import datetime, date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from app.config import settings
from app.models import (
    Base, Pedagang, Lokasi, Penjualan, Kunjungan, Episode, QTableEntry, RuteOptimal,
    SesiPenjualan
)


class DatabaseManager:
    """
    DatabaseManager class handles all database operations.
    Methods: getLokasi(), getPenjualan(), saveEpisode(), saveRuteOptimal()
    """
    
    def __init__(self):
        """Initialize database connection."""
        self.engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_db(self):
        """Dependency for FastAPI endpoints."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    # ==================== Pedagang Operations ====================
    
    def get_pedagang_by_username(self, db: Session, username: str) -> Optional[Pedagang]:
        """Get pedagang by username."""
        return db.query(Pedagang).filter(Pedagang.username == username).first()
    
    def get_pedagang_by_id(self, db: Session, pedagang_id: int) -> Optional[Pedagang]:
        """Get pedagang by ID."""
        return db.query(Pedagang).filter(Pedagang.id == pedagang_id).first()
    
    def create_pedagang(self, db: Session, nama: str, username: str, hashed_password: str) -> Pedagang:
        """Create a new pedagang."""
        pedagang = Pedagang(
            nama=nama,
            username=username,
            password=hashed_password
        )
        db.add(pedagang)
        db.commit()
        db.refresh(pedagang)
        return pedagang
    
    # ==================== Lokasi Operations ====================
    
    def get_lokasi(self, db: Session, pedagang_id: int) -> List[Lokasi]:
        """
        Get all locations for a pedagang.
        Implements getLokasi() from Class Diagram.
        """
        return db.query(Lokasi).filter(Lokasi.pedagang_id == pedagang_id).all()
    
    def get_lokasi_by_id(self, db: Session, lokasi_id: int) -> Optional[Lokasi]:
        """Get a single location by ID."""
        return db.query(Lokasi).filter(Lokasi.id == lokasi_id).first()
    
    def create_lokasi(
        self, db: Session, pedagang_id: int, nama: str, latitude: float, longitude: float
    ) -> Lokasi:
        """Create a new location."""
        lokasi = Lokasi(
            pedagang_id=pedagang_id,
            nama=nama,
            latitude=latitude,
            longitude=longitude
        )
        db.add(lokasi)
        db.commit()
        db.refresh(lokasi)
        return lokasi
    
    def update_lokasi(
        self, db: Session, lokasi_id: int, nama: str, latitude: float, longitude: float
    ) -> Optional[Lokasi]:
        """Update an existing location."""
        lokasi = db.query(Lokasi).filter(Lokasi.id == lokasi_id).first()
        if lokasi:
            lokasi.nama = nama
            lokasi.latitude = latitude
            lokasi.longitude = longitude
            db.commit()
            db.refresh(lokasi)
        return lokasi
    
    def delete_lokasi(self, db: Session, lokasi_id: int) -> bool:
        """Delete a location."""
        lokasi = db.query(Lokasi).filter(Lokasi.id == lokasi_id).first()
        if lokasi:
            db.delete(lokasi)
            db.commit()
            return True
        return False
    
    # ==================== Penjualan Operations ====================
    
    def get_penjualan(
        self,
        db: Session,
        pedagang_id: int,
        limit: int = 20,
        offset: int = 0,
        lokasi_id: int = None,
        hari_kuliah: int = None
    ) -> List[Penjualan]:
        
        query = db.query(Penjualan).filter(
            Penjualan.pedagang_id == pedagang_id
        )

        if lokasi_id is not None:
            query = query.filter(
                Penjualan.lokasi_id == lokasi_id
            )
        
        if hari_kuliah is not None:
            query = query.filter(
                Penjualan.hari_kuliah == hari_kuliah
            )

        return query.order_by(
            Penjualan.waktu_kunjungan.desc()
        ).offset(offset).limit(limit).all()
    
    def get_penjualan_by_lokasi(
        self, db: Session, pedagang_id: int, lokasi_id: int
    ) -> List[Penjualan]:
        """Get sales records for a specific location."""
        return db.query(Penjualan).filter(
            Penjualan.pedagang_id == pedagang_id,
            Penjualan.lokasi_id == lokasi_id
        ).all()
    
    def create_penjualan(
        self, 
        db: Session, 
        pedagang_id: int, 
        lokasi_id: int,
        jumlah_terjual: int,
        kondisi_cuaca: str = "cerah", 
        hari_kuliah: int = 1,
        kunjungan_id: Optional[int] = None
    ) -> Penjualan:
        """
        Create a new sales record.
        - Jika kunjungan_id diberikan: simpan sebagai transaksi individual per konsumen.
        - Jika tidak: legacy mode — akumulasi per hari untuk backward compat Q-Learning.
        """
        from sqlalchemy import func
        now = datetime.utcnow()

        if kunjungan_id is not None:
            # Mode baru: transaksi individual terhubung ke kunjungan
            penjualan = Penjualan(
                pedagang_id=pedagang_id,
                lokasi_id=lokasi_id,
                kunjungan_id=kunjungan_id,
                jumlah_terjual=jumlah_terjual,
                kondisi_cuaca=kondisi_cuaca,
                hari_kuliah=hari_kuliah,
                durasi_mangkal=0.0,
                waktu_transaksi=now,
                waktu_kunjungan=now
            )
            db.add(penjualan)
            db.commit()
            db.refresh(penjualan)
            return penjualan
        else:
            # Mode legacy: cari/akumulasi record harian per lokasi
            today = now.date()
            existing = db.query(Penjualan).filter(
                Penjualan.pedagang_id == pedagang_id,
                Penjualan.lokasi_id == lokasi_id,
                Penjualan.kunjungan_id == None,  # noqa: E711
                func.date(Penjualan.waktu_kunjungan) == today
            ).first()

            if existing:
                existing.jumlah_terjual += jumlah_terjual
                time_diff = now - existing.waktu_kunjungan
                existing.durasi_mangkal = int(time_diff.total_seconds() / 60)
                db.commit()
                db.refresh(existing)
                return existing
            else:
                penjualan = Penjualan(
                    pedagang_id=pedagang_id,
                    lokasi_id=lokasi_id,
                    kunjungan_id=None,
                    jumlah_terjual=jumlah_terjual,
                    kondisi_cuaca=kondisi_cuaca,
                    durasi_mangkal=0,
                    hari_kuliah=hari_kuliah,
                    waktu_kunjungan=now,
                    waktu_transaksi=now
                )
                db.add(penjualan)
                db.commit()
                db.refresh(penjualan)
                return penjualan

    
    def get_performa_penjualan(self, db: Session, pedagang_id: int, days: int = 7):
        """
        Performa penjualan per hari — agregasi dari semua transaksi konsumen.
        
        jumlah_terjual berisi nominal belanja per konsumen (rupiah),
        sehingga SUM(jumlah_terjual) = total pendapatan hari itu.
        """
        from sqlalchemy import func
        
        results = db.query(
            func.date(Penjualan.waktu_kunjungan).label('tanggal'),
            func.sum(Penjualan.jumlah_terjual).label('total_pendapatan'),
            func.count(Penjualan.id).label('total_transaksi')
        ).filter(
            Penjualan.pedagang_id == pedagang_id
        ).group_by(
            func.date(Penjualan.waktu_kunjungan)
        ).order_by(
            func.date(Penjualan.waktu_kunjungan).desc()
        ).limit(days).all()
        
        # Balik urutan agar chronological (terlama → terbaru)
        results.reverse()
        
        return results
    
    # ==================== Kunjungan Operations ====================

    def get_active_kunjungan(
        self, db: Session, pedagang_id: int
    ) -> Optional[Kunjungan]:
        """Mendapatkan kunjungan yang sedang aktif (waktu_selesai IS NULL)."""
        return db.query(Kunjungan).filter(
            Kunjungan.pedagang_id == pedagang_id,
            Kunjungan.waktu_selesai == None  # noqa: E711
        ).order_by(Kunjungan.waktu_mulai.desc()).first()

    def create_kunjungan(
        self,
        db: Session,
        pedagang_id: int,
        lokasi_id: int,
        kondisi_cuaca: str = "cerah",
        hari_kuliah: int = 1
    ) -> Kunjungan:
        """Membuat kunjungan baru (sesi mangkal di lokasi tertentu)."""
        kunjungan = Kunjungan(
            pedagang_id=pedagang_id,
            lokasi_id=lokasi_id,
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

    def close_kunjungan(
        self, db: Session, kunjungan: Kunjungan
    ) -> Kunjungan:
        """
        Menutup kunjungan aktif:
        - Set waktu_selesai = sekarang
        - Hitung durasi_mangkal otomatis dalam JAM (float)
        """
        now = datetime.utcnow()
        kunjungan.waktu_selesai = now
        delta = now - kunjungan.waktu_mulai
        durasi_jam = round(delta.total_seconds() / 3600, 2)
        kunjungan.durasi_mangkal = max(0.02, durasi_jam)
        db.commit()
        db.refresh(kunjungan)
        return kunjungan

    def get_kunjungan_list(
        self,
        db: Session,
        pedagang_id: int,
        limit: int = 20,
        offset: int = 0,
        lokasi_id: Optional[int] = None,
        hari_kuliah: Optional[int] = None,
        hanya_selesai: bool = False,
        hanya_aktif: bool = False
    ) -> List[Kunjungan]:
        """Mengambil list kunjungan dengan berbagai filter."""
        query = db.query(Kunjungan).filter(
            Kunjungan.pedagang_id == pedagang_id
        )
        if lokasi_id is not None:
            query = query.filter(Kunjungan.lokasi_id == lokasi_id)
        if hari_kuliah is not None:
            query = query.filter(Kunjungan.hari_kuliah == hari_kuliah)
        if hanya_selesai:
            query = query.filter(Kunjungan.waktu_selesai != None)  # noqa: E711
        if hanya_aktif:
            query = query.filter(Kunjungan.waktu_selesai == None)  # noqa: E711
        return query.order_by(
            Kunjungan.waktu_mulai.desc()
        ).offset(offset).limit(limit).all()

    def get_kunjungan_by_id(
        self, db: Session, kunjungan_id: int
    ) -> Optional[Kunjungan]:
        """Mendapatkan kunjungan berdasarkan ID."""
        return db.query(Kunjungan).filter(Kunjungan.id == kunjungan_id).first()

    # ==================== Episode Operations ====================
    
    def save_episode(
        self, db: Session, pedagang_id: int, total_reward: float, total_waktu: int
    ) -> Episode:
        """
        Save a training episode.
        Implements saveEpisode() from Class Diagram.
        """
        episode = Episode(
            pedagang_id=pedagang_id,
            total_reward=total_reward,
            total_waktu=total_waktu
        )
        db.add(episode)
        db.commit()
        db.refresh(episode)
        return episode
    
    def get_episodes(self, db: Session, pedagang_id: int) -> List[Episode]:
        """Get all episodes for a pedagang."""
        return db.query(Episode).filter(
            Episode.pedagang_id == pedagang_id
        ).order_by(Episode.created_at.desc()).all()
    
    # ==================== Q-Table Operations ====================
    
    def get_q_value(
        self, db: Session, pedagang_id: int, state_key: str, action: int
    ) -> float:
        """Get Q-value for a state-action pair."""
        entry = db.query(QTableEntry).filter(
            QTableEntry.pedagang_id == pedagang_id,
            QTableEntry.state_key == state_key,
            QTableEntry.action == action
        ).first()
        return entry.q_value if entry else 0.0
    
    def update_q_value(
        self, db: Session, pedagang_id: int, state_key: str, action: int, q_value: float
    ):
        """Update Q-value for a state-action pair."""
        entry = db.query(QTableEntry).filter(
            QTableEntry.pedagang_id == pedagang_id,
            QTableEntry.state_key == state_key,
            QTableEntry.action == action
        ).first()
        
        if entry:
            entry.q_value = q_value
            entry.updated_at = datetime.utcnow()
        else:
            entry = QTableEntry(
                pedagang_id=pedagang_id,
                state_key=state_key,
                action=action,
                q_value=q_value
            )
            db.add(entry)
        db.commit()
    
    def get_all_q_values(self, db: Session, pedagang_id: int) -> List[QTableEntry]:
        """Get all Q-table entries for a pedagang."""
        return db.query(QTableEntry).filter(
            QTableEntry.pedagang_id == pedagang_id
        ).all()
    
    # ==================== Rute Optimal Operations ====================
    
    def save_rute_optimal(
        self, db: Session, pedagang_id: int, urutan_lokasi: list,
        total_jarak: float, total_reward: float, rekomendasi: str
    ) -> RuteOptimal:
        """
        Save optimal route.
        Implements saveRuteOptimal() from Class Diagram.
        """
        # Delete old optimal route for this pedagang
        db.query(RuteOptimal).filter(
            RuteOptimal.pedagang_id == pedagang_id
        ).delete()
        
        rute = RuteOptimal(
            pedagang_id=pedagang_id,
            urutan_lokasi=urutan_lokasi,
            total_jarak=total_jarak,
            total_reward=total_reward,
            rekomendasi=rekomendasi
        )
        db.add(rute)
        db.commit()
        db.refresh(rute)
        return rute
    
    def get_rute_optimal(self, db: Session, pedagang_id: int) -> Optional[RuteOptimal]:
        """Get the latest optimal route for a pedagang."""
        return db.query(RuteOptimal).filter(
            RuteOptimal.pedagang_id == pedagang_id
        ).order_by(RuteOptimal.created_at.desc()).first()

    # ==================== Sesi Penjualan Operations ====================

    def get_active_sesi(self, db: Session, pedagang_id: int) -> Optional[SesiPenjualan]:
        """Mendapatkan sesi penjualan yang sedang aktif (waktu_selesai IS NULL)."""
        return db.query(SesiPenjualan).filter(
            SesiPenjualan.pedagang_id == pedagang_id,
            SesiPenjualan.waktu_selesai == None  # noqa: E711
        ).order_by(SesiPenjualan.waktu_mulai.desc()).first()

    def create_sesi_penjualan(
        self,
        db: Session,
        pedagang_id: int,
        kondisi_cuaca: str = "cerah",
        hari_kuliah: int = 1,
        stok_awal: Optional[int] = None
    ) -> SesiPenjualan:
        """Membuat sesi penjualan baru (Start Berjualan)."""
        sesi = SesiPenjualan(
            pedagang_id=pedagang_id,
            waktu_mulai=datetime.utcnow(),
            waktu_selesai=None,
            stok_awal=stok_awal,
            kondisi_cuaca=kondisi_cuaca,
            hari_kuliah=hari_kuliah
        )
        db.add(sesi)
        db.commit()
        db.refresh(sesi)
        return sesi

    def close_sesi_penjualan(
        self,
        db: Session,
        sesi: SesiPenjualan
    ) -> SesiPenjualan:
        """
        Menutup sesi penjualan (Stop Berjualan):
        - Set waktu_selesai = sekarang
        - Hitung durasi total
        - Hitung total pendapatan & transaksi dari semua kunjungan
        - Tutup kunjungan aktif jika ada
        """
        from sqlalchemy import func

        now = datetime.utcnow()
        sesi.waktu_selesai = now

        # Hitung durasi total dalam jam
        delta = now - sesi.waktu_mulai
        sesi.durasi_total = round(delta.total_seconds() / 3600, 2)

        # Tutup kunjungan aktif jika masih ada
        active_kunjungan = db.query(Kunjungan).filter(
            Kunjungan.pedagang_id == sesi.pedagang_id,
            Kunjungan.sesi_id == sesi.id,
            Kunjungan.waktu_selesai == None  # noqa: E711
        ).all()

        for k in active_kunjungan:
            k.waktu_selesai = now
            delta_k = now - k.waktu_mulai
            durasi_jam = round(delta_k.total_seconds() / 3600, 2)
            k.durasi_mangkal = max(0.02, durasi_jam)

        # Hitung ringkasan dari semua kunjungan dalam sesi ini
        kunjungan_in_sesi = db.query(Kunjungan).filter(
            Kunjungan.sesi_id == sesi.id
        ).all()

        total_pendapatan = 0
        total_transaksi = 0
        lokasi_ids = set()

        for k in kunjungan_in_sesi:
            lokasi_ids.add(k.lokasi_id)
            # Hitung total transaksi dalam kunjungan ini
            transaksi_list = db.query(Penjualan).filter(
                Penjualan.kunjungan_id == k.id
            ).all()
            for t in transaksi_list:
                total_pendapatan += t.jumlah_terjual
                total_transaksi += 1

        sesi.total_pendapatan = total_pendapatan
        sesi.total_transaksi = total_transaksi
        sesi.total_lokasi_dikunjungi = len(lokasi_ids)

        db.commit()
        db.refresh(sesi)
        return sesi

    def get_kunjungan_by_sesi(
        self, db: Session, sesi_id: int
    ) -> List[Kunjungan]:
        """Mengambil semua kunjungan dalam satu sesi."""
        return db.query(Kunjungan).filter(
            Kunjungan.sesi_id == sesi_id
        ).order_by(Kunjungan.waktu_mulai.asc()).all()


# Global database manager instance
db_manager = DatabaseManager()
