package com.safescreen.ai.database

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Database
import androidx.room.RoomDatabase

@Entity(tableName = "security_logs")
data class SecurityLogEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long,
    val threatType: String, // "NSFW", "DEEPFAKE", "SCAM_OCR", "PII_REDACTION"
    val confidenceScore: Float,
    val actionTaken: String, // "BLURRED", "BLOCKED", "WARNED", "REDACTED"
    val activeAppPackage: String
)

@Dao
interface SecurityLogDao {
    @Insert
    suspend fun insertLog(log: SecurityLogEntity)

    @Query("SELECT * FROM security_logs ORDER BY timestamp DESC LIMIT 100")
    suspend fun getRecentLogs(): List<SecurityLogEntity>

    @Query("DELETE FROM security_logs")
    suspend fun clearLogs()
}

@Database(entities = [SecurityLogEntity::class], version = 1, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun securityLogDao(): SecurityLogDao
}
