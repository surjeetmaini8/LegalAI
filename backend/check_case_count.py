
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient
from app.core.config import settings


def check_cases():
    """Check current case count in MongoDB"""
    try:
        # Use settings from your config
        if not settings.MONGODB_URL:
            raise ValueError(
                "MONGODB_URL environment variable is required. Please set it in your .env file."
            )
        mongo_uri = settings.MONGODB_URL
        db_name = (
            settings.MONGODB_DATABASE if settings.MONGODB_DATABASE else "dcm_system"
        )

        
        client = MongoClient(mongo_uri)
        db = client[db_name]

        # Get case count
        total_cases = db.cases.count_documents({})

       
        if total_cases > 0:
            # Get breakdown by case type
            print("🔹 Breakdown by Case Type:")
            for case_type in [
                "CRIMINAL",
                "CIVIL",
                "FAMILY",
                "COMMERCIAL",
                "CONSTITUTIONAL",
            ]:
                count = db.cases.count_documents({"case_type": case_type})
                if count > 0:
                    print(f"   • {case_type}: {count}")

            # Get breakdown by priority
            print("\n🔹 Breakdown by Priority:")
            for priority in ["URGENT", "HIGH", "MEDIUM", "LOW"]:
                count = db.cases.count_documents({"priority": priority})
                if count > 0:
                    print(f"   • {priority}: {count}")

            # Get breakdown by status
            print("\n🔹 Breakdown by Status:")
            for status in ["FILED", "UNDER_REVIEW", "SCHEDULED", "HEARING", "DISPOSED"]:
                count = db.cases.count_documents({"status": status})
                if count > 0:
                    print(f"   • {status}: {count}")
        else:
            print(" No cases found in database!")
           
        client.close()
        return total_cases

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()
        return -1


if __name__ == "__main__":
    count = check_cases()
    print("\n")
