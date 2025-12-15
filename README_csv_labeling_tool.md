# CSV Labeling Tool (Flask)

## Start
1. Install dependencies:
   pip install flask pandas

2. Run:
   python csv_labeling_tool_app.py

3. Open:
   http://127.0.0.1:5000

## Notes
- The currently loaded CSV is stored at: ./data/current.csv (relative to the script)
- Selecting a label auto-saves it into the selected coder column
- Resume: opens at the first unlabeled row for that coder; if complete, shows a Thank-you page
- You can download the CSV anytime from the UI
