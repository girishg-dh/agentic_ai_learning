from fastapi import FastAPI
from main import run_university_admissions_crew

app = FastAPI(
    title="University Admission Assistant API",
    description="An API to get a detailed report on TU9 university admission requirements.",
    version='1.0.0'
)


@app.post("/get-admissions-report", summary="Generate a university admissions repor")
async def get_report():
    """
    Triggers the University Admissions Crew to generate a comprehensive report
    on TU9 universities for a Berlin Abitur student.
    """
    print("Received request to generate admissions report...")
    final_report = run_university_admissions_crew()
    return {"report": final_report}


@app.get("/", summary="Check API status")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "University Admissions Assistant API is running."}