"""
Module 24: User Feedback Loop
Records and manages user feedback actions (Accept, Reject, Click, Ignore)
and provides an interface for local feedback inspection and model retraining.
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger("Feedback")


class FeedbackManager:
    """Manages local user feedback storage and retrieval."""

    def __init__(self, feedback_csv: str = "outputs/feedback.csv"):
        self.feedback_path = Path(feedback_csv)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.feedback_path.exists():
            df_init = pd.DataFrame(columns=[
                "user_id", "product_id", "recommendation_rank", "action", "timestamp"
            ])
            df_init.to_csv(self.feedback_path, index=False)

    def record_feedback(self, user_id: int, product_id: int, rank: int, action: str) -> dict:
        """
        Appends an interaction feedback record.
        Actions: 'Accept', 'Reject', 'Click', 'Ignore'
        """
        valid_actions = ["Accept", "Reject", "Click", "Ignore"]
        if action not in valid_actions:
            raise ValueError(f"Action '{action}' not recognized. Must be one of {valid_actions}")

        record = {
            "user_id": int(user_id),
            "product_id": int(product_id),
            "recommendation_rank": int(rank),
            "action": action,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        df = pd.read_csv(self.feedback_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(self.feedback_path, index=False)
        logger.info(f"Feedback recorded: User {user_id}, Prod {product_id}, Action {action}")
        return record

    def get_all_feedback(self) -> pd.DataFrame:
        """Returns full historical feedback dataframe."""
        if self.feedback_path.exists():
            return pd.read_csv(self.feedback_path)
        return pd.DataFrame(columns=["user_id", "product_id", "recommendation_rank", "action", "timestamp"])

    def get_action_counts(self) -> dict:
        """Returns aggregate action counts."""
        df = self.get_all_feedback()
        if df.empty:
            return {"Accept": 0, "Reject": 0, "Click": 0, "Ignore": 0}
        counts = df["action"].value_counts().to_dict()
        for act in ["Accept", "Reject", "Click", "Ignore"]:
            counts.setdefault(act, 0)
        return counts


if __name__ == "__main__":
    fb = FeedbackManager()
    fb.record_feedback(1024, 844, 1, "Accept")
    fb.record_feedback(1024, 865, 2, "Click")
    fb.record_feedback(1056, 851, 1, "Reject")
    print("Feedback summary:", fb.get_action_counts())
    print("\nRecent feedback:\n", fb.get_all_feedback().tail())
