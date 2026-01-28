"""
FastAPI backend for EDA application
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import uuid
from typing import Optional
import asyncio

from analyzer import analyze_dataset
from llm_service import LLMService
from plot_generator import PlotGenerator
from models import AnalysisJob, JobStatus

# ===========================================
# 1. CREATE THE APP FIRST
# ===========================================
app = FastAPI(
    title="EDA Insights API",
    description="Automated Exploratory Data Analysis",
    version="1.0.0"
)

# ===========================================
# 2. ADD CORS MIDDLEWARE
# ===========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================
# 3. MOUNT FRONTEND (AFTER app is created!)
# ===========================================
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ===========================================
# 4. STORAGE AND SERVICES
# ===========================================
jobs: dict[str, AnalysisJob] = {}
dataframes: dict[str, pd.DataFrame] = {}

llm_service = LLMService()
plot_generator = PlotGenerator()


# ===========================================
# 5. FRONTEND ROUTES
# ===========================================
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found. Use /docs for API documentation."}


@app.get("/styles.css")
async def serve_css():
    """Serve CSS file"""
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
async def serve_js():
    """Serve JavaScript file"""
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


# ===========================================
# 6. API ROUTES
# ===========================================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    schema: Optional[UploadFile] = File(None)
):
    """Upload a CSV file for analysis"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    job_id = str(uuid.uuid4())
    
    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
    
    dataframes[job_id] = df
    
    schema_content = None
    if schema:
        schema_content = (await schema.read()).decode('utf-8')
    
    jobs[job_id] = AnalysisJob(
        job_id=job_id,
        filename=file.filename,
        status=JobStatus.PROCESSING,
        progress=0
    )
    
    background_tasks.add_task(run_analysis, job_id, df, file.filename, schema_content)
    
    return {"job_id": job_id, "message": "Analysis started"}


async def run_analysis(job_id: str, df: pd.DataFrame, filename: str, schema_content: Optional[str]):
    """Run the complete analysis pipeline"""
    job = jobs[job_id]
    
    try:
        # Stage 1: Basic analysis (0-40%)
        job.progress = 10
        await asyncio.sleep(0.1)
        
        analysis = analyze_dataset(df)
        analysis["dataset_info"]["filename"] = filename
        
        job.progress = 40
        await asyncio.sleep(0.1)
        
        # Stage 2: Generate plots (40-60%)
        job.progress = 50
        plots = plot_generator.generate_auto_plots(df, analysis["columns"])
        analysis["plots"] = plots
        
        job.progress = 60
        await asyncio.sleep(0.1)
        
        # Stage 3: LLM analysis (60-90%)
        job.progress = 70
        llm_analysis = await llm_service.analyze(
            analysis["dataset_info"],
            analysis["columns"],
            schema_content
        )
        analysis["llm_analysis"] = llm_analysis
        
        job.progress = 90
        await asyncio.sleep(0.1)
        
        # Complete
        job.progress = 100
        job.status = JobStatus.COMPLETED
        job.results = analysis
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        print(f"Analysis failed: {e}")


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get the status of an analysis job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get the results of a completed analysis"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Analysis still in progress")
    
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error)
    
    return job.results


@app.post("/api/suggest-plot")
async def suggest_plot(request: dict):
    """Get LLM suggestion for plot type"""
    job_id = request.get("job_id")
    columns = request.get("columns", [])
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if not job.results:
        raise HTTPException(status_code=400, detail="No analysis results")
    
    col_info = [
        col for col in job.results["columns"] 
        if col["name"] in columns
    ]
    
    suggestion = await llm_service.suggest_plot(col_info)
    return suggestion


@app.post("/api/generate-plot")
async def generate_plot(request: dict):
    """Generate a custom plot"""
    job_id = request.get("job_id")
    plot_type = request.get("plot_type")
    x_column = request.get("x_column")
    y_column = request.get("y_column")
    color_column = request.get("color_column")
    columns = request.get("columns", [])
    
    if job_id not in dataframes:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = dataframes[job_id]
    
    plot_data = plot_generator.generate_custom_plot(
        df, plot_type, x_column, y_column, color_column, columns
    )
    
    return plot_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)