import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import os

# Default fallback rules if Firebase is not reachable
DEFAULT_RULES = {
    "min_wall_thickness": 2.5, # Aluminum Die Casting
    "required_draft_angle": 3.0, # Degrees
    "max_rib_thickness_ratio": 0.6 # Ratio to nominal wall
}

class RuleEngine:
    def __init__(self, key_path=None, db_url=None):
        """
        Initializes connection to Firebase Realtime Database.
        If credentials/DB URL are not provided or invalid, uses default rules.
        """
        self.connected = False
        self.rules = DEFAULT_RULES.copy()
        
        # In a real enterprise app, these would come from env vars
        key_path = key_path or os.getenv("FIREBASE_KEY_PATH", "serviceAccountKey.json")
        db_url = db_url or os.getenv("FIREBASE_DB_URL", "https://varroc-ai-default-rtdb.firebaseio.com/")
        
        if os.path.exists(key_path):
            try:
                # Check if default app is already initialized
                if not firebase_admin._apps:
                    cred = credentials.Certificate(key_path)
                    firebase_admin.initialize_app(cred, {
                        'databaseURL': db_url
                    })
                self.connected = True
                print("Firebase connected successfully.")
            except Exception as e:
                print(f"Failed to initialize Firebase: {e}. Falling back to default rules.")
        else:
            print("Firebase credentials not found. Operating with local default rules.")
            
    def fetch_rules(self):
        """Fetches the latest rule set from Firebase."""
        if self.connected:
            try:
                ref = db.reference('design_rules')
                remote_rules = ref.get()
                if remote_rules:
                    # Update local cache
                    self.rules.update(remote_rules)
            except Exception as e:
                print(f"Error fetching from Firebase: {e}")
                
        return self.rules
        
    def check_draft_angle(self, actual_angle, rule_override=None):
        """Hard constraint check for draft angles."""
        required = rule_override if rule_override is not None else self.rules["required_draft_angle"]
        if actual_angle < required:
            return False, f"FAIL: Draft angle {actual_angle:.1f} is less than required {required:.1f}"
        return True, "PASS"

if __name__ == "__main__":
    # Test execution
    engine = RuleEngine()
    current_rules = engine.fetch_rules()
    print("Active Rules:", current_rules)
    
    # Test check
    is_valid, msg = engine.check_draft_angle(1.5)
    print("Draft Check 1.5 deg:", msg)
    is_valid, msg = engine.check_draft_angle(3.5)
    print("Draft Check 3.5 deg:", msg)
