from tbparse import SummaryReader
import os

reader = SummaryReader("runs")
df = reader.scalars

os.makedirs("history", exist_ok=True)
df.to_csv("history/mixup_history.csv", index=False)
