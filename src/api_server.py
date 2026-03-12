# src/api_server.py
from flask import Flask, jsonify, request
import sqlite3
import os
from datetime import datetime, timedelta

app = Flask(__name__)
DB_PATH = "../db/attendance.db"


@app.route("/attendance", methods=["GET"])
def get_attendance():
    """Get all attendance records"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Optional filters
        name_filter = request.args.get('name')
        date_filter = request.args.get('date')

        query = "SELECT * FROM attendance WHERE 1=1"
        params = []

        if name_filter:
            query += " AND name = ?"
            params.append(name_filter)

        if date_filter:
            query += " AND DATE(timestamp) = ?"
            params.append(date_filter)

        query += " ORDER BY timestamp DESC"

        rows = cursor.execute(query, params).fetchall()
        conn.close()

        attendance_data = []
        for row in rows:
            attendance_data.append({
                "id": row[0],
                "name": row[1],
                "timestamp": row[2]
            })

        return jsonify({
            "status": "success",
            "count": len(attendance_data),
            "data": attendance_data
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/attendance/stats", methods=["GET"])
def get_stats():
    """Get attendance statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Total records
        total = cursor.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]

        # Unique persons
        unique = cursor.execute("SELECT COUNT(DISTINCT name) FROM attendance").fetchone()[0]

        # Today's count
        today = cursor.execute(
            "SELECT COUNT(*) FROM attendance WHERE DATE(timestamp) = DATE('now')"
        ).fetchone()[0]

        conn.close()

        return jsonify({
            "status": "success",
            "stats": {
                "total_records": total,
                "unique_persons": unique,
                "today_attendance": today
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)