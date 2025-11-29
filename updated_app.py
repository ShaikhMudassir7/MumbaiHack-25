import os
import fitz  # PyMuPDF
import pytesseract
from pdfminer.high_level import extract_text as pdfminer_extract
import pdfplumber
import json
import base64
import requests
from flask import Flask, render_template, request, jsonify, g, send_file, redirect, url_for
from PIL import Image
import io
import tempfile
import logging
import logging.handlers
from datetime import datetime
import traceback
import time
import psutil
from werkzeug.utils import secure_filename
from typing import Dict, Any, Tuple, Optional, List
from dotenv import load_dotenv
import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
import re
import asyncio
import concurrent.futures
from pathlib import Path
import uuid
import numpy as np
from io import BytesIO
import xlsxwriter

# --- add imports (if not present) ---
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import os

# --- concurrency limit for LLM requests (tweak via env var) ---
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "4"))
llm_semaphore = threading.Semaphore(LLM_CONCURRENCY)

# --- resilient requests session factory ---
def create_retry_session(retries: int = 5, backoff_factor: float = 1.0,
                         status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Default headers (you can change to 'close' if keep-alive causes issue)
    session.headers.update({
        "Connection": "keep-alive", 
        "Accept": "application/json"
    })
    return session

# single shared session (thread-safe usage for requests)
LLM_SESSION = create_retry_session()


# Load environment variables
load_dotenv()

# Import your OCI LLM modules with error handling
try:
    from ocimeta_llm import meta_llm_response
    from ocicohere_llm import cohere_llm_response
    from ocimetavision_llm import vision_llm_response
    OCI_AVAILABLE = True
except ImportError as e:
    print(f"OCI modules not available: {e}")
    OCI_AVAILABLE = False

# Enhanced Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.request_count = 0
        self.total_response_time = 0
        self.error_count = 0
        self.start_time = datetime.now()
        self.batch_stats = {
            'total_batches': 0,
            'total_files_processed': 0,
            'total_invoices_extracted': 0,
            'successful_extractions': 0,
            'failed_extractions': 0
        }
    
    def before_request(self):
        g.start_time = time.time()
        self.request_count += 1
    
    def after_request(self, response):
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            self.total_response_time += response_time
            
            if response.status_code >= 400:
                self.error_count += 1
        
        return response
    
    def update_batch_stats(self, files_processed: int, invoices_found: int, successful: int, failed: int):
        self.batch_stats['total_batches'] += 1
        self.batch_stats['total_files_processed'] += files_processed
        self.batch_stats['total_invoices_extracted'] += invoices_found
        self.batch_stats['successful_extractions'] += successful
        self.batch_stats['failed_extractions'] += failed
    
    def get_stats(self):
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        uptime = datetime.now() - self.start_time
        
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'avg_response_time': round(avg_response_time, 3),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'uptime_hours': round(uptime.total_seconds() / 3600, 2),
            'oci_available': OCI_AVAILABLE,
            'batch_stats': self.batch_stats
        }

# Initialize Flask app
def create_app(config_name='production'):
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
    app.config['BATCH_FOLDER'] = os.getenv('BATCH_FOLDER', 'batch_uploads')
    app.config['EXPORT_FOLDER'] = os.getenv('EXPORT_FOLDER', 'exports')
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    app.config['MAX_BATCH_FILES'] = int(os.getenv('MAX_BATCH_FILES', 50))
    
    # Create directories
    for folder in [app.config['UPLOAD_FOLDER'], app.config['BATCH_FOLDER'], app.config['EXPORT_FOLDER']]:
        os.makedirs(folder, exist_ok=True)
    
    # Security headers
    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        if not app.debug:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response
    
    return app

app = create_app()

# Initialize performance monitor
monitor = PerformanceMonitor()
app.before_request(monitor.before_request)
app.after_request(monitor.after_request)

# Global variables to store extracted data
extracted_invoice_data = {}
current_extracted_text = ""
batch_processing_results = []
invoice_database = {}

# Dashboard global variables
all_jobs_data = {}
job_master = None
DEFAULT_COMPANY = "JAMES L WILLIAMS ME LIMITED"

# Multi-invoice detection patterns
INVOICE_PATTERNS = [
    r'(?i)(?:tax\s+)?invoice(?:\s+(?:no|number|#))?[:.\s]*([A-Z0-9\-]+)',
    r'(?i)bill(?:\s+(?:no|number|#))?[:.\s]*([A-Z0-9\-]+)',
    r'(?i)receipt(?:\s+(?:no|number|#))?[:.\s]*([A-Z0-9\-]+)',
]

PAGE_BREAK_PATTERNS = [
    r'(?i)page\s+\d+\s+of\s+\d+',
    r'(?i)continued\s+on\s+next\s+page',
    r'(?i)end\s+of\s+invoice',
    r'(?i)new\s+invoice',
    r'(?i)invoice\s+total',
]

def detect_multiple_invoices(text: str) -> List[Dict[str, Any]]:
    """Detect multiple invoices in a single PDF text"""
    invoices = []
    
    # Split text by potential invoice boundaries
    sections = []
    current_section = ""
    lines = text.split('\n')
    
    for line in lines:
        # Check for invoice start patterns
        invoice_match = False
        for pattern in INVOICE_PATTERNS:
            if re.search(pattern, line):
                if current_section.strip():
                    sections.append(current_section)
                current_section = line + '\n'
                invoice_match = True
                break
        
        if not invoice_match:
            current_section += line + '\n'
    
    if current_section.strip():
        sections.append(current_section)
    
    # Process each potential invoice section
    for i, section in enumerate(sections):
        if len(section.strip()) > 100:  # Minimum length check
            invoices.append({
                'invoice_number': i + 1,
                'text': section,
                'estimated_start': section[:200] + '...' if len(section) > 200 else section
            })
    
    return invoices if len(invoices) > 1 else [{'invoice_number': 1, 'text': text, 'estimated_start': text[:200] + '...'}]

@app.route('/dashboard/batch/<batch_id>')
def dashboard_batch(batch_id):
    """Dashboard for batch of invoices - NEW"""
    # Check if batch exists
    batch_exists = any(
        inv.get('processing_metadata', {}).get('batch_id') == batch_id 
        for inv in invoice_database.values()
    )
    
    if not batch_exists:
        global extracted_invoice_data
        if not extracted_invoice_data or extracted_invoice_data.get('batch_id') != batch_id:
            return redirect(url_for('index'))
    
    html_file = 'main.html'
    if os.path.exists(html_file):
        return send_file(html_file)
    else:
        return jsonify({"error": "Dashboard template not found"}), 404

def setup_logging():
    """Enhanced logging setup"""
    if not app.debug:
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Application logs
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'invoice_processor.log'),
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        
        # Error logs
        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'errors.log'),
            maxBytes=10485760,
            backupCount=5
        )
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        error_handler.setLevel(logging.ERROR)
        
        app.logger.addHandler(file_handler)
        app.logger.addHandler(error_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Enhanced Invoice Processor started')

setup_logging()

# Configure Tesseract - Fix path detection
tesseract_cmd = os.getenv('TESSERACT_CMD')
if not tesseract_cmd:
    # Try common Tesseract paths
    common_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', ''))
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            tesseract_cmd = path
            break

if tesseract_cmd and os.path.exists(tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
else:
    print("Warning: Tesseract not found. OCR functionality will be limited.")

# OCI Configuration
OCI_CONFIG = {
    'compartment_id': os.getenv('OCI_COMPARTMENT_ID'),
    'model_ocids': {
        "cohere": os.getenv('OCI_COHERE_MODEL'),
        "llama-3.1": os.getenv('OCI_LLAMA_31_MODEL'),
        "llama-3.2": os.getenv('OCI_LLAMA_32_MODEL'),
        "llama-3.3": os.getenv('OCI_LLAMA_33_MODEL'),  
        "llama-vision": os.getenv('OCI_LLAMA_VISION_MODEL'),
        "grok-3": os.getenv('OCI_GROK_3_MODEL'),
        "grok-4": os.getenv('OCI_GROK_4_MODEL')
    }
}

# Mistral API configuration - Updated with better error handling
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY', 'MSGfE03w0EaQrXH2sVI9pnvftCyMqWyZ')
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

if not MISTRAL_API_KEY:
    print("Warning: MISTRAL_API_KEY not found in environment variables. Mistral LLM will not be available.")

# Error handlers
@app.errorhandler(413)
def too_large(e):
    app.logger.warning(f"File too large uploaded from {request.remote_addr}")
    return jsonify({"error": "File too large. Maximum size is 16MB.", "status": "error"}), 413

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error occurred", "status": "error"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404

@app.errorhandler(429)
def rate_limit_exceeded(e):
    app.logger.warning(f"Rate limit exceeded from {request.remote_addr}")
    return jsonify({"error": "Rate limit exceeded. Please try again later.", "status": "error"}), 429

# Routes
@app.route('/')
def index():
    """Main route - serves the invoice processor"""
    html_file = 'index.html'
    if os.path.exists(html_file):
        return send_file(html_file)
    else:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Invoice Processing System</title>
        </head>
        <body>
            <h1>Invoice Processing System</h1>
            <p>Please ensure the enhanced_invoice_processor.html file is in the same directory as app.py</p>
        </body>
        </html>
        '''

@app.route('/dashboard')
def dashboard_main():
    """Main dashboard showing all jobs"""
    html_file = 'main.html'
    if os.path.exists(html_file):
        return send_file(html_file)
    else:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard</title>
        </head>
        <body>
            <h1>Project Dashboard</h1>
            <p>main.html file not found</p>
        </body>
        </html>
        '''

@app.route('/dashboard/<job_number>')
def dashboard_job(job_number):
    """Dashboard for specific job"""
    html_file = 'dashboard.html'
    if os.path.exists(html_file):
        return send_file(html_file)
    else:
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard - {job_number}</title>
        </head>
        <body>
            <h1>Job Dashboard - {job_number}</h1>
            <p>dashboard.html file not found</p>
        </body>
        </html>
        '''

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle single file upload and processing - UPDATED"""
    start_time = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part", "status": "error"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file", "status": "error"}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported", "status": "error"}), 400
        
        llm_model = request.form.get('llm_model', 'mistral')
        ocr_model = request.form.get('ocr_model', 'auto')
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}_{filename}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        app.logger.info(f"Processing file: {filename} with LLM: {llm_model}, OCR: {ocr_model}")
        
        # Process PDF
        preview_img = generate_pdf_preview(file_path)
        extracted_text, ocr_method = extract_text_with_selected_ocr(file_path, ocr_model)
        
        global current_extracted_text
        current_extracted_text = extracted_text
        
        # Detect multiple invoices
        detected_invoices = detect_multiple_invoices(extracted_text)
        
        # Process all detected invoices with enhanced data structure
        processed_invoices = []
        for idx, invoice in enumerate(detected_invoices):
            structured_data = process_with_selected_llm(invoice['text'], llm_model, file_path)
            
            invoice_id = f"{timestamp}_{unique_id}_{idx+1}"
            processed_invoice = {
                'invoice_id': invoice_id,
                'invoice_number': invoice.get('invoice_number', idx + 1),
                'data': structured_data,
                'status': 'success',
                'extracted_text': invoice['text'],
                'processing_metadata': {
                    'llm_model': llm_model,
                    'ocr_method': ocr_method,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'source_file': file.filename
                }
            }
            processed_invoices.append(processed_invoice)
            
            # Store in database for dashboard access
            invoice_database[invoice_id] = processed_invoice
        
        # Update global storage
        global extracted_invoice_data
        batch_id = f"BATCH_{timestamp}_{unique_id}"
        extracted_invoice_data = {
            'invoices': processed_invoices,
            'count': len(processed_invoices),
            'extraction_timestamp': datetime.now().isoformat(),
            'source': 'single',
            'batch_id': batch_id,
            'source_filename': file.filename
        }
        
        # Clean up file
        try:
            os.unlink(file_path)
        except Exception as cleanup_error:
            app.logger.warning(f"Failed to cleanup file {file_path}: {cleanup_error}")
        
        processing_time = round(time.time() - start_time, 2)
        app.logger.info(f"Successfully processed file: {filename} in {processing_time}s")
        
        # Determine dashboard route
        is_multiple = len(processed_invoices) > 1
        if is_multiple:
            dashboard_route = f"/dashboard/batch/{batch_id}"
        else:
            dashboard_route = f"/dashboard/invoice/{processed_invoices[0]['invoice_id']}"
        
        response_data = {
            "preview": preview_img,
            "text": extracted_text[:2000],
            "json_data": processed_invoices[0]['data'] if processed_invoices else {},
            "ocr_method": ocr_method,
            "llm_used": llm_model,
            "ocr_selected": ocr_model,
            "processing_time": processing_time,
            "invoices_count": len(processed_invoices),
            "status": "success",
            "show_dashboard": True,
            "dashboard_type": "multiple" if is_multiple else "single",
            "dashboard_route": dashboard_route,
            "dashboard_url": dashboard_route,
            "batch_id": batch_id
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"Error processing upload: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
            
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/batch-upload', methods=['POST'])
def batch_upload():
    """Handle batch file upload and processing - UPDATED"""
    start_time = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part", "status": "error"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file", "status": "error"}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported", "status": "error"}), 400
        
        llm_model = request.form.get('llm_model', 'mistral')
        ocr_model = request.form.get('ocr_model', 'auto')
        split_mode = request.form.get('split_mode', 'auto')
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        batch_id = f"BATCH_{timestamp}_{unique_id}"
        filename = f"{timestamp}_{unique_id}_{filename}"
        
        file_path = os.path.join(app.config['BATCH_FOLDER'], filename)
        file.save(file_path)
        
        app.logger.info(f"Processing batch file: {filename} with mode: {split_mode}")
        
        extracted_text, ocr_method = extract_text_with_selected_ocr(file_path, ocr_model)
        
        # Detect multiple invoices if needed
        invoices = []
        if split_mode in ['multi', 'auto']:
            detected_invoices = detect_multiple_invoices(extracted_text)
            if len(detected_invoices) > 1 or split_mode == 'multi':
                invoices = detected_invoices
            else:
                invoices = [{'invoice_number': 1, 'text': extracted_text, 'estimated_start': extracted_text[:200] + '...'}]
        else:
            invoices = [{'invoice_number': 1, 'text': extracted_text, 'estimated_start': extracted_text[:200] + '...'}]
        
        # Process each invoice with enhanced data structure
        processed_invoices = []
        for idx, invoice in enumerate(invoices):
            try:
                structured_data = process_with_selected_llm(invoice['text'], llm_model, file_path)
                
                invoice_id = f"{batch_id}_{idx+1}"
                processed_invoice = {
                    'invoice_id': invoice_id,
                    'invoice_number': invoice.get('invoice_number', idx + 1),
                    'data': structured_data,
                    'status': 'success',
                    'text_preview': invoice['estimated_start'],
                    'extracted_text': invoice['text'],
                    'processing_metadata': {
                        'llm_model': llm_model,
                        'ocr_method': ocr_method,
                        'extraction_timestamp': datetime.now().isoformat(),
                        'source_file': file.filename,
                        'batch_id': batch_id
                    }
                }
                processed_invoices.append(processed_invoice)
                
                # Store in database
                invoice_database[invoice_id] = processed_invoice
                
            except Exception as e:
                app.logger.error(f"Error processing invoice {invoice['invoice_number']}: {str(e)}")
                processed_invoices.append({
                    'invoice_id': f"{batch_id}_{idx+1}",
                    'invoice_number': invoice['invoice_number'],
                    'error': str(e),
                    'status': 'error',
                    'text_preview': invoice['estimated_start']
                })
        
        # Clean up file
        try:
            os.unlink(file_path)
        except Exception as cleanup_error:
            app.logger.warning(f"Failed to cleanup batch file {file_path}: {cleanup_error}")
        
        processing_time = round(time.time() - start_time, 2)
        successful_invoices = len([inv for inv in processed_invoices if inv['status'] == 'success'])
        
        app.logger.info(f"Batch processed: {filename}, {len(invoices)} invoices, {successful_invoices} successful, {processing_time}s")
        
        # Update monitoring stats
        monitor.update_batch_stats(1, len(invoices), successful_invoices, len(invoices) - successful_invoices)
        
        # Store for dashboard generation
        global extracted_invoice_data
        extracted_invoice_data = {
            'invoices': processed_invoices,
            'count': len(processed_invoices),
            'extraction_timestamp': datetime.now().isoformat(),
            'source': 'batch',
            'batch_id': batch_id,
            'source_filename': file.filename
        }
        
        # Determine dashboard route
        dashboard_route = f"/dashboard/batch/{batch_id}"
        
        response_data = {
            "filename": file.filename,
            "invoices_found": len(invoices),
            "invoices_processed": len(processed_invoices),
            "successful_extractions": successful_invoices,
            "json_data": processed_invoices[0]['data'] if processed_invoices and processed_invoices[0]['status'] == 'success' else {},
            "ocr_method": ocr_method,
            "llm_used": llm_model,
            "split_mode": split_mode,
            "processing_time": processing_time,
            "status": "success",
            "show_dashboard": True,
            "dashboard_type": "multiple" if len(processed_invoices) > 1 else "single",
            "dashboard_route": dashboard_route,
            "dashboard_url": dashboard_route,
            "batch_id": batch_id,
            "invoices_count": len(processed_invoices)
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        app.logger.error(f"Error processing batch upload: {str(e)}")
        app.logger.error(traceback.format_exc())
        
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
            
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route('/api/invoice-dashboard-data')
def get_single_invoice_dashboard_data():
    """Get dashboard data for single extracted invoice"""
    global extracted_invoice_data
    
    try:
        if not extracted_invoice_data or 'invoices' not in extracted_invoice_data:
            return jsonify({'error': 'No invoice data available. Please upload an invoice first.'}), 404
        
        if len(extracted_invoice_data['invoices']) == 0:
            return jsonify({'error': 'No invoices found in extracted data'}), 404
        
        invoice = extracted_invoice_data['invoices'][0]
        data = invoice.get('data', {})
        
        # Extract and calculate values with defaults
        line_items = data.get('Line_Items', [])
        total_amount = float(data.get('Total_Amount', 0))
        gst_amount = float(data.get('GST_Amount', 0))
        amount_excl_gst = float(data.get('Amount_excld_GST', 0))
        freight = float(data.get('Freight_Packing_Handling_chrgs', 0))
        
        dashboard_data = {
            'invoice_number': data.get('Invoice_Number', 'N/A'),
            'invoice_date': data.get('Invoice_Date', 'N/A'),
            'document_type': data.get('Document_Type', 'INV'),
            'vendor_name': data.get('Store_ID', 'Unknown Vendor'),
            'po_number': data.get('PO_Number', 'N/A'),
            'abn_number': data.get('ABN_Number', 'N/A'),
            'total_amount': total_amount,
            'gst_amount': gst_amount,
            'amount_excl_gst': amount_excl_gst,
            'freight_charges': freight,
            'line_items_count': len(line_items),
            'line_items': line_items,
            'extraction_timestamp': extracted_invoice_data.get('extraction_timestamp', datetime.now().isoformat())
        }
        
        return jsonify(dashboard_data)
    
    except Exception as e:
        app.logger.error(f"Error in get_single_invoice_dashboard_data: {str(e)}")
        return jsonify({'error': f'Failed to load invoice data: {str(e)}'}), 500


@app.route('/dashboard/invoice/<invoice_id>')
def dashboard_single_invoice(invoice_id):
    """Dashboard for single invoice - UPDATED"""
    # Check if invoice exists
    if invoice_id not in invoice_database:
        global extracted_invoice_data
        if not extracted_invoice_data or extracted_invoice_data.get('count', 0) == 0:
            return redirect(url_for('index'))
    
    html_file = 'dashboard.html'
    if os.path.exists(html_file):
        return send_file(html_file)
    else:
        return jsonify({"error": "Dashboard template not found"}), 404


@app.route('/api/jobs-summary')
def get_jobs_summary():
    """Get summary of all jobs sorted by overbudget amount"""
    try:
        # For now, return mock data - in production, this would load from your data source
        jobs_summary = [
            {
                'job_number': 'JOB001',
                'job_name': 'Construction Project A',
                'total_budget': 1500000.0,
                'total_actual': 1650000.0,
                'total_variance': -150000.0,
                'over_running': 150000.0,
                'under_running': 0.0,
                'budget_utilization': 110.0,
                'record_count': 45,
                'critical_risks': 3,
                'status': 'Overrun'
            },
            {
                'job_number': 'JOB002', 
                'job_name': 'Infrastructure Project B',
                'total_budget': 2000000.0,
                'total_actual': 1800000.0,
                'total_variance': 200000.0,
                'over_running': 0.0,
                'under_running': 200000.0,
                'budget_utilization': 90.0,
                'record_count': 38,
                'critical_risks': 1,
                'status': 'Within Budget'
            }
        ]
        
        # Sort by over-running descending
        jobs_summary.sort(key=lambda x: x['over_running'], reverse=True)
        
        return jsonify(jobs_summary)
    
    except Exception as e:
        print(f"Error getting jobs summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_number>/kpi_data')
def get_job_kpi_data(job_number):
    """Get KPI data for specific job"""
    try:
        # Mock data - replace with actual data loading logic
        kpi_data = {
            'job_number': job_number,
            'total_budget': 1500000.0,
            'total_actual': 1650000.0,
            'total_variance': -150000.0,
            'over_running': 150000.0,
            'under_running': 0.0,
            'risk_distribution': {
                'Critical': 3,
                'Medium': 8,
                'Low': 12,
                'No Budget': 2
            },
            'record_count': 45
        }
        
        return jsonify(kpi_data)
    
    except Exception as e:
        print(f"Error getting KPI data for {job_number}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_number>/chart_data')
def get_job_chart_data(job_number):
    """Get chart data for specific job"""
    try:
        # Mock chart data
        chart_data = {
            'budget_vs_actual': {
                'accounts': ['Concrete', 'Steel', 'Labor', 'Equipment', 'Materials'],
                'budget': [500000, 400000, 350000, 200000, 150000],
                'actual': [550000, 450000, 400000, 180000, 170000]
            },
            'risk_distribution': {
                'labels': ['Low', 'Medium', 'Critical', 'No Budget'],
                'data': [12, 8, 3, 2],
                'accounts': {
                    'Low': ['Office Supplies', 'Travel', 'Training'],
                    'Medium': ['Equipment Rental', 'Subcontractors'],
                    'Critical': ['Concrete', 'Steel', 'Labor'],
                    'No Budget': ['Miscellaneous', 'Contingency']
                },
                'matrix': {
                    'Negligible': {'Improbable': 2, 'Remote': 1, 'Occasional': 0, 'Probable': 0, 'Frequent': 0},
                    'Low': {'Improbable': 3, 'Remote': 2, 'Occasional': 1, 'Probable': 0, 'Frequent': 0},
                    'Moderate': {'Improbable': 1, 'Remote': 3, 'Occasional': 2, 'Probable': 1, 'Frequent': 0},
                    'Significant': {'Improbable': 0, 'Remote': 1, 'Occasional': 2, 'Probable': 1, 'Frequent': 0},
                    'Catastrophic': {'Improbable': 0, 'Remote': 0, 'Occasional': 1, 'Probable': 2, 'Frequent': 0}
                }
            },
            'variance_analysis': {
                'accounts': ['Concrete', 'Steel', 'Labor', 'Equipment', 'Materials', 'Subcontractors'],
                'variances': [-50000, -50000, -50000, 20000, -20000, 10000]
            }
        }
        
        return jsonify(chart_data)
    
    except Exception as e:
        print(f"Error getting chart data for {job_number}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_number>/forecast_data')
def get_job_forecast_data(job_number):
    """Get forecast data for specific job"""
    try:
        # Mock forecast data
        forecast_data = {
            'months': ['Current', 'Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'],
            'current_actual': [1650000, None, None, None, None, None, None],
            'forecasted_actual': [1650000, 1720000, 1800000, 1880000, 1960000, 2040000, 2120000]
        }
        
        return jsonify(forecast_data)
    
    except Exception as e:
        print(f"Error getting forecast data for {job_number}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_number>/drilldown_data')
def get_job_drilldown_data(job_number):
    """Get drilldown data for specific job"""
    try:
        # Mock drilldown data
        drilldown_data = [
            {
                'Cost Code': 'CONC001',
                'Account Description': 'Concrete Supply',
                'Revised Budget Amt': 500000,
                'Actual Amount': 550000,
                'Budget Var Amount': -50000,
                'Risk_Factor': 'Critical',
                'Budget_Status': 'Overrun',
                'Percent Complete': 110
            },
            {
                'Cost Code': 'STEEL001',
                'Account Description': 'Steel Materials', 
                'Revised Budget Amt': 400000,
                'Actual Amount': 450000,
                'Budget Var Amount': -50000,
                'Risk_Factor': 'Critical',
                'Budget_Status': 'Overrun',
                'Percent Complete': 112
            }
        ]
        
        return jsonify(drilldown_data)
    
    except Exception as e:
        print(f"Error getting drilldown data for {job_number}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_number>/additional_chart_data')
def get_additional_chart_data(job_number):
    """Get additional chart data"""
    try:
        # Mock additional chart data
        additional_data = {
            'top_variance': [
                {'account': 'Concrete Supply', 'variance': -50000},
                {'account': 'Steel Materials', 'variance': -50000},
                {'account': 'Labor Costs', 'variance': -50000},
                {'account': 'Equipment Rental', 'variance': 20000},
                {'account': 'Material Supply', 'variance': -20000}
            ],
            'budget_percentile': {
                'accounts': ['Concrete', 'Steel', 'Labor', 'Equipment', 'Materials', 'Subcontractors'],
                'percentiles': [110, 112, 114, 90, 113, 95]
            }
        }
        
        return jsonify(additional_data)
    
    except Exception as e:
        print(f"Error getting additional chart data for {job_number}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_insight', methods=['POST'])
def get_ai_insight():
    """Get AI insights for chart data"""
    try:
        data = request.json
        chart_type = data.get('chart_type', '')
        
        # Mock AI insights
        insights = {
            'budget_vs_actual': "The budget vs actual analysis shows significant overruns in concrete, steel, and labor costs. Concrete is over budget by 10%, steel by 12.5%, and labor by 14.3%. Equipment costs are under budget by 10%, which partially offsets the overruns. Immediate attention needed for the top three overrunning categories.",
            'risk_distribution': "Risk analysis indicates 3 critical risk items primarily in core construction materials. The project has 8 medium-risk accounts that require monitoring. Consider implementing cost control measures for critical items and weekly reviews for medium-risk categories.",
            'variance_analysis': "Variance analysis reveals consistent overruns across major cost categories. The top 3 variances account for 75% of total overrun. Recommendation: Review supplier contracts and labor efficiency for concrete, steel, and labor categories.",
            'forecast': "Based on current spending trends, the project is forecasted to exceed budget by 28% if current patterns continue. The burn rate indicates completion in 5 months with significant cost overruns. Immediate cost containment measures recommended.",
            'top_variance': "The top 5 variance accounts represent 92% of total budget variance. Concrete, steel, and labor show the most significant deviations. These three categories alone account for 75% of the overrun. Strategic focus on these areas could yield 80% of cost savings.",
            'budget_percentile': "Budget completion analysis shows 4 accounts exceeding 110% completion, indicating severe overruns. Labor costs are at 114% completion - highest among all categories. Equipment is the only category under 100% completion. Resource reallocation from equipment to overrunning categories may help balance the budget."
        }
        
        insight = insights.get(chart_type, "AI insight not available for this chart type.")
        
        return jsonify({
            'success': True,
            'insight': f"<div style='color: black;'><ul><li>{insight}</li></ul></div>"
        })
        
    except Exception as e:
        print(f"Error generating AI insight: {e}")
        return jsonify({
            'success': False,
            'insight': f'Unable to generate AI insight: {str(e)}'
        })

@app.route('/api/export_chart_data', methods=['POST'])
def export_chart_data():
    """Export chart data to Excel"""
    try:
        data = request.json
        chart_type = data.get('chart_type', '')
        
        output = BytesIO()
        workbook = xlsxwriter.Workbook(output)
        worksheet = workbook.add_worksheet(chart_type)
        
        # Add some sample data
        worksheet.write(0, 0, f"Exported data for {chart_type}")
        worksheet.write(1, 0, "Generated on:")
        worksheet.write(1, 1, datetime.now().isoformat())
        
        workbook.close()
        output.seek(0)
        
        return jsonify({
            'success': True,
            'data': output.getvalue().hex()
        })
        
    except Exception as e:
        print(f"Error exporting data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/job/<job_number>/approval_request', methods=['POST'])
def handle_job_approval_request(job_number):
    """Handle approval requests for critical items"""
    try:
        data = request.json
        cost_code = data.get('cost_code')
        account_desc = data.get('account_description')
        variance = data.get('variance')
        
        # Log the approval request
        print(f"=== APPROVAL REQUEST ===")
        print(f"Job Number: {job_number}")
        print(f"Account: {account_desc}")
        print(f"Cost Code: {cost_code}")
        print(f"Variance: ${variance:,.2f}")
        print(f"========================")
        
        request_id = f"APR-{job_number}-{cost_code}-{int(datetime.now().timestamp())}"
        
        return jsonify({
            'success': True,
            'message': 'Approval request submitted successfully',
            'request_id': request_id,
            'job_number': job_number
        })
        
    except Exception as e:
        print(f"Error processing approval request for job {job_number}: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# Existing invoice processing routes and functions...
@app.route('/export-excel', methods=['POST'])
def export_excel():
    """Export processed data to Excel format"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided", "status": "error"}), 400
        
        batch_results = data.get('data', [])
        settings = data.get('settings', {})
        
        if not batch_results:
            return jsonify({"error": "No batch results to export", "status": "error"}), 400
        
        # Create Excel file
        excel_path = create_excel_export(batch_results, settings)
        
        app.logger.info(f"Excel export created: {excel_path}")
        
        return send_file(
            excel_path,
            as_attachment=True,
            download_name=f"invoice_batch_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    except Exception as e:
        app.logger.error(f"Excel export error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/query', methods=['POST'])
def query_invoice():
    """Handle user prompts about the extracted invoice data"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided", "status": "error"}), 400
        
        user_prompt = data.get('prompt', '').strip()
        llm_model = data.get('llm_model', 'mistral')
        
        if not user_prompt:
            return jsonify({"error": "Prompt is required", "status": "error"}), 400
        
        if len(user_prompt) > 4000:
            return jsonify({"error": "Prompt too long. Maximum 1000 characters.", "status": "error"}), 400
        
        if not current_extracted_text:
            return jsonify({"error": "No invoice data available. Please upload an invoice first.", "status": "error"}), 400
        
        app.logger.info(f"Processing query: {user_prompt[:100]}... with LLM: {llm_model}")
        
        # Process user query with selected LLM
        response = process_user_query(user_prompt, llm_model)
        
        processing_time = round(time.time() - start_time, 2)
        app.logger.info(f"Successfully processed user query in {processing_time}s")
        
        return jsonify({
            "response": response,
            "llm_used": llm_model,
            "processing_time": processing_time,
            "status": "success"
        })
    
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint"""
    try:
        stats = monitor.get_stats()
        
        # Check system resources
        cpu_ok = stats['cpu_percent'] < 90
        memory_ok = stats['memory_percent'] < 90
        
        # Check disk space
        try:
            disk_usage = psutil.disk_usage('/')
            disk_ok = (disk_usage.free / disk_usage.total) > 0.1  # More than 10% free
        except:
            disk_ok = True  # Assume OK if we can't check
        
        # Check API connectivity
        api_status = "not_configured"
        if MISTRAL_API_KEY:
            api_status = "configured"
        
        health_status = "healthy" if (cpu_ok and memory_ok and disk_ok) else "degraded"
        
        return jsonify({
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0",
            "system": {
                "cpu_ok": cpu_ok,
                "memory_ok": memory_ok,
                "disk_ok": disk_ok,
                "oci_available": OCI_AVAILABLE,
                "mistral_api_configured": bool(MISTRAL_API_KEY),
                "tesseract_available": bool(tesseract_cmd and os.path.exists(tesseract_cmd))
            },
            "stats": stats,
            "api_status": api_status,
            "features": {
                "batch_processing": True,
                "multi_invoice_detection": True,
                "excel_export": True,
                "advanced_ocr": True,
                "multiple_llm_support": True,
                "dashboard_integration": True
            }
        })
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500
    

# Ensure the multi-invoice dashboard route works
@app.route('/dashboard/multi')
def dashboard_multiple_invoices():
    """Dashboard for multiple invoices extracted from single file"""
    global extracted_invoice_data
    
    if not extracted_invoice_data or extracted_invoice_data.get('count', 0) == 0:
        # Redirect to upload page if no data
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>No Data Available</title>
            <script>
                setTimeout(function() {
                    window.location.href = '/';
                }, 2000);
            </script>
        </head>
        <body>
            <div style="text-align: center; padding: 50px;">
                <h2>No invoice data available</h2>
                <p>Redirecting to upload page...</p>
                <p><a href="/">Click here if not redirected</a></p>
            </div>
        </body>
        </html>
        '''
    
    # Serve main.html for multiple invoices
    html_file = 'main.html'
    if os.path.exists(html_file):
        return send_file(html_file)
    else:
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>Multiple Invoices Dashboard</title></head>
        <body>
            <h1>Multiple Invoices Dashboard</h1>
            <p>main.html file not found</p>
        </body>
        </html>
        '''
    
@app.route('/api/extracted-invoice-data')
def get_extracted_invoice_data():
    """API endpoint to get extracted invoice data for dashboard"""
    global extracted_invoice_data
    
    if not extracted_invoice_data:
        return jsonify({'error': 'No invoice data available'}), 404
    
    return jsonify(extracted_invoice_data)

@app.route('/api/invoice/<invoice_id>/dashboard-data')
def get_invoice_dashboard_data(invoice_id):
    """Get dashboard data for specific extracted invoice - ENHANCED"""
    try:
        # Check invoice database first
        if invoice_id in invoice_database:
            invoice_data = invoice_database[invoice_id]
            data = invoice_data['data']
        else:
            # Fallback to global extracted data
            global extracted_invoice_data
            if not extracted_invoice_data or 'invoices' not in extracted_invoice_data:
                return jsonify({'error': 'No invoice data available'}), 404
            
            # Find invoice by ID
            invoice_data = None
            for inv in extracted_invoice_data['invoices']:
                if inv.get('invoice_id') == invoice_id:
                    invoice_data = inv
                    break
            
            if not invoice_data:
                return jsonify({'error': 'Invoice not found'}), 404
            
            data = invoice_data['data']
        
        # Extract and calculate values with proper handling
        line_items = data.get('Line_Items', [])
        total_amount = safe_float(data.get('Total_Amount', 0))
        gst_amount = safe_float(data.get('GST_Amount', 0))
        amount_excl_gst = safe_float(data.get('Amount_excld_GST', 0))
        freight = safe_float(data.get('Freight_Packing_Handling_chrgs', 0))
        
        # Calculate if GST not provided
        if gst_amount == 0 and total_amount > 0 and amount_excl_gst > 0:
            gst_amount = total_amount - amount_excl_gst
        
        # Enhanced dashboard data structure
        dashboard_data = {
            'invoice_id': invoice_id,
            'invoice_number': data.get('Invoice_Number', 'N/A'),
            'invoice_date': data.get('Invoice_Date', 'N/A'),
            'document_type': data.get('Document_Type', 'INV'),
            'vendor_name': data.get('Store_ID', 'Unknown Vendor'),
            'po_number': data.get('PO_Number', 'N/A'),
            'abn_number': data.get('ABN_Number', 'N/A'),
            'total_amount': total_amount,
            'gst_amount': gst_amount,
            'amount_excl_gst': amount_excl_gst,
            'freight_charges': freight,
            'line_items_count': len(line_items),
            'line_items': line_items,
            'extraction_timestamp': invoice_data.get('processing_metadata', {}).get('extraction_timestamp', datetime.now().isoformat()),
            'processing_metadata': invoice_data.get('processing_metadata', {}),
            # Chart data
            'chart_data': prepare_invoice_chart_data(data, line_items)
        }
        
        return jsonify(dashboard_data)
    
    except Exception as e:
        app.logger.error(f"Error in get_invoice_dashboard_data: {str(e)}")
        return jsonify({'error': f'Failed to load invoice data: {str(e)}'}), 500

@app.route('/api/batch/<batch_id>/dashboard-data')
def get_batch_dashboard_data(batch_id):
    """Get dashboard data for batch of invoices - NEW"""
    try:
        # Find all invoices for this batch
        batch_invoices = [inv for inv_id, inv in invoice_database.items() 
                         if inv.get('processing_metadata', {}).get('batch_id') == batch_id]
        
        if not batch_invoices:
            # Fallback to global data
            global extracted_invoice_data
            if extracted_invoice_data and extracted_invoice_data.get('batch_id') == batch_id:
                batch_invoices = extracted_invoice_data.get('invoices', [])
            else:
                return jsonify({'error': 'Batch not found'}), 404
        
        # Prepare summary data
        jobs_summary = []
        for invoice in batch_invoices:
            if invoice.get('status') != 'success':
                continue
                
            data = invoice.get('data', {})
            
            total_amount = safe_float(data.get('Total_Amount', 0))
            amount_excl_gst = safe_float(data.get('Amount_excld_GST', 0))
            gst_amount = safe_float(data.get('GST_Amount', 0))
            
            if gst_amount == 0 and total_amount > 0 and amount_excl_gst > 0:
                gst_amount = total_amount - amount_excl_gst
            
            line_items = data.get('Line_Items', [])
            
            jobs_summary.append({
                'invoice_id': invoice.get('invoice_id'),
                'job_number': data.get('Invoice_Number', f'INV-{invoice.get("invoice_number", "?")}'),
                'job_name': data.get('Store_ID', 'Unknown Vendor'),
                'invoice_date': data.get('Invoice_Date', 'N/A'),
                'po_number': data.get('PO_Number', 'N/A'),
                'total_budget': total_amount,
                'total_actual': amount_excl_gst,
                'total_variance': gst_amount,
                'over_running': 0,
                'under_running': 0,
                'budget_utilization': 100,
                'record_count': len(line_items),
                'critical_risks': 0,
                'status': data.get('Document_Type', 'INV')
            })
        
        return jsonify(jobs_summary)
    
    except Exception as e:
        app.logger.error(f"Error in get_batch_dashboard_data: {str(e)}")
        return jsonify({'error': f'Failed to load batch data: {str(e)}'}), 500

@app.route('/api/multi-invoice-dashboard-data')
def get_multi_invoice_dashboard_data():
    """Get dashboard data for multiple extracted invoices - ENHANCED"""
    global extracted_invoice_data
    
    try:
        if not extracted_invoice_data or 'invoices' not in extracted_invoice_data:
            return jsonify({'error': 'No invoice data available. Please upload invoices first.'}), 404
        
        invoices = extracted_invoice_data['invoices']
        
        if len(invoices) == 0:
            return jsonify({'error': 'No invoices found in extracted data'}), 404
        
        jobs_summary = []
        for invoice in invoices:
            if invoice.get('status') != 'success':
                continue
                
            data = invoice.get('data', {})
            
            total_amount = safe_float(data.get('Total_Amount', 0))
            amount_excl_gst = safe_float(data.get('Amount_excld_GST', 0))
            gst_amount = safe_float(data.get('GST_Amount', 0))
            
            if gst_amount == 0 and total_amount > 0 and amount_excl_gst > 0:
                gst_amount = total_amount - amount_excl_gst
            
            line_items = data.get('Line_Items', [])
            
            jobs_summary.append({
                'invoice_id': invoice.get('invoice_id'),
                'job_number': data.get('Invoice_Number', f'INV-{invoice.get("invoice_number", "?")}'),
                'job_name': data.get('Store_ID', 'Unknown Vendor'),
                'invoice_date': data.get('Invoice_Date', 'N/A'),
                'po_number': data.get('PO_Number', 'N/A'),
                'total_budget': total_amount,
                'total_actual': amount_excl_gst,
                'total_variance': gst_amount,
                'over_running': 0,
                'under_running': 0,
                'budget_utilization': 100,
                'record_count': len(line_items),
                'critical_risks': 0,
                'status': data.get('Document_Type', 'INV')
            })
        
        return jsonify(jobs_summary)
    
    except Exception as e:
        app.logger.error(f"Error in get_multi_invoice_dashboard_data: {str(e)}")
        return jsonify({'error': f'Failed to load invoice data: {str(e)}'}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Enhanced metrics endpoint for monitoring"""
    try:
        stats = monitor.get_stats()
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Metrics endpoint failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Utility functions (from original app.py)
def generate_pdf_preview(pdf_path: str) -> str:
    """Generate base64 encoded preview of the first page"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Optimize image size
        img.thumbnail((800, 1000), Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        doc.close()
        
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        app.logger.error(f"Error generating PDF preview: {str(e)}")
        raise

def extract_text_with_selected_ocr(pdf_path: str, ocr_model: str) -> Tuple[str, str]:
    """Extract text using selected OCR engine"""
    try:
        if ocr_model == "auto":
            return extract_text_with_ocr(pdf_path)
        
        elif ocr_model == "pymupdf":
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if len(text.strip()) > 10:
                return text, "PyMuPDF"
            else:
                raise RuntimeError("PyMuPDF extracted insufficient text")
        
        elif ocr_model == "pdfplumber":
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            if len(text.strip()) > 10:
                return text, "PDFPlumber"
            else:
                raise RuntimeError("PDFPlumber extracted insufficient text")
        
        elif ocr_model == "pdfminer":
            text = pdfminer_extract(pdf_path)
            if len(text.strip()) > 10:
                return text, "PDFMiner"
            else:
                raise RuntimeError("PDFMiner extracted insufficient text")
        
        elif ocr_model == "tesseract":
            if not tesseract_cmd:
                raise RuntimeError("Tesseract not available")
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
            doc.close()
            if len(text.strip()) > 10:
                return text, "Tesseract OCR"
            else:
                raise RuntimeError("Tesseract OCR extracted insufficient text")
        
        elif ocr_model == "tesseract_enhanced":
            if not tesseract_cmd:
                raise RuntimeError("Tesseract not available")
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()/-:$%@#&*+=[]{}|\\;"'
                text += pytesseract.image_to_string(img, config=custom_config)
            doc.close()
            
            if len(text.strip()) > 10:
                return text, "Tesseract OCR (Enhanced)"
            else:
                raise RuntimeError("Enhanced Tesseract OCR extracted insufficient text")
        
        else:
            raise RuntimeError(f"Unknown OCR model: {ocr_model}")
            
    except Exception as e:
        app.logger.error(f"OCR extraction failed: {str(e)}")
        raise

def extract_text_with_ocr(pdf_path: str) -> Tuple[str, str]:
    """Extract text using multiple OCR engines with fallback"""
    strategies = [
        ("PyMuPDF", lambda: extract_with_pymupdf(pdf_path)),
        ("PDFPlumber", lambda: extract_with_pdfplumber(pdf_path)),
        ("PDFMiner", lambda: extract_with_pdfminer(pdf_path)),
    ]
    
    # Add Tesseract if available
    if tesseract_cmd:
        strategies.append(("Tesseract OCR", lambda: extract_with_tesseract(pdf_path)))
    
    for method_name, extraction_func in strategies:
        try:
            text = extraction_func()
            if len(text.strip()) > 50:
                return text, method_name
        except Exception as e:
            app.logger.warning(f"{method_name} failed: {str(e)}")
            continue
    
    raise RuntimeError("All OCR methods failed")

def extract_with_tesseract(pdf_path: str) -> str:
    """Extract text using Tesseract OCR"""
    if not tesseract_cmd:
        raise RuntimeError("Tesseract not available")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    doc.close()
    return text

def extract_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using PDFPlumber"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_with_pdfminer(pdf_path: str) -> str:
    """Extract text using PDFMiner"""
    return pdfminer_extract(pdf_path)

def extract_with_pymupdf(pdf_path: str) -> str:
    """Extract text using PyMuPDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def get_invoice_prompt() -> str:
    """Get the standard invoice extraction prompt"""
    return """
    INSTRUCTION:
    You are an intelligent document information extractor. Your task is to extract structured invoice data from raw OCR-extracted text from supplier invoices.

    The OCR content may be slightly noisy and unstructured, but you should use context clues and keyword matching to accurately identify and extract information.

    Use the following fields for extraction:
     
    HEADER FIELDS IN INVOICE:
    ABN,Date,Tax Invoice NO,Your reference,Header sold block line 2,Not labeled, ,Not labeled,Tax Invoice Total
     
    The value of PO_Number is equal to the value of 'Your Reference' 
    The value of Store_Id is equal to the second line value of 'TAX INVOICE' like this PARRAMATTA CARPET COURT.
    The value of Invoice_Number is equal to value of 'Tax Invoice No' like this 2446445.
     
    LINE ITEM FIELDS (array of objects) IN INVOICE:
    Each line item should include: Product,Description,Quantity Dlevrd,U/M,Unit Price,Extended
     
    Respond only with a JSON structure. If data is missing or ambiguous, return null for that field.
     
    Capture Header Field keys in JSON as:
    Document_Type, ABN_Number, Invoice_Date, Invoice_Number, PO_Number, Store_ID, Amount_excld_GST, Freight_Packing_Handling_chrgs, GST_Amount, Total_Amount
     
    Document_Type can be either 'INV' (Invoice) or 'CN' (Credit Note).
    If key words like Credit/Tax Adjustment Note is present in ocr extracted text then give Document_Type as 'CN' otherwise 'INV'.
    **ALWAYS** Give Invoice_Date field value in YYYY-MM-DD format strictly.
    If there is no Freight Charges found in text then give its value as '0.0' in output.
     
    Capture Line Item Field keys in JSON as:
    Product_Code, Product_Name, Quantity, UoM, Unit_Price and Line_Total_Amount. Make sure to capture all line items.
    If Quantity or Unit_Price or Line_Total_Amount are blank or empty, default it to 0.
     
    Extract Shipping, Handling, Transport, Freight etc Charges as Product_code and Product_Name and its charges as unit price in line item
     
    Give the output in JSON format ONLY, do not explain or include any other text. Do not include any currency symbol like $.
    Ensure keys are named exactly as provided above with underscore in between words instead of space, even if named differently in the invoice.
     
    ENCLOSE OUTPUT STRICTLY BETWEEN:
    ### START ###
    <JSON>
    ### END ###
     
    Here is the OCR content:
    """

def get_query_prompt(user_query: str) -> str:
    """Get prompt for user queries"""
    return f"""
    You are an intelligent invoice data assistant. You have access to extracted invoice data and raw OCR text.
    
    INSTRUCTIONS:
    1. Answer the user's question based on the invoice data and raw text provided below, if question is not related to invoice respond "Not found in extracted information".
    2. Provide accurate, specific information from the invoice.
    3. If the requested information is not available, clearly state that it's not found.
    4. Always respond in valid JSON format with the following structure:
    {{
        "answer": "your detailed answer here",
        "data_found": true/false,
        "extracted_fields": {{
            "field_name": "field_value",
            ...
        }},
        "confidence": "high/medium/low"
    }}
    
    5. For questions about amounts, dates, or specific fields, extract the exact values.
    6. Be conversational but precise in your answers.
    
    USER QUESTION: {user_query}
    
    EXTRACTED INVOICE DATA:
    {json.dumps(extracted_invoice_data, indent=2) if extracted_invoice_data else "No structured data available"}
    
    RAW OCR TEXT:
    {current_extracted_text[:3000]}...
    
    Please provide your response in the JSON format specified above.
    """

def process_user_query(user_prompt: str, llm_model: str) -> Dict[str, Any]:
    """Process user query using selected LLM"""
    try:
        if OCI_AVAILABLE and llm_model in OCI_CONFIG['model_ocids'] and OCI_CONFIG['model_ocids'][llm_model]:
            return process_query_with_oci_llm(user_prompt, llm_model)
        else:
            return process_query_with_mistral(user_prompt)
    except Exception as e:
        app.logger.error(f"Query processing failed: {str(e)}")
        return {
            "answer": f"Error processing query: {str(e)}",
            "data_found": False,
            "extracted_fields": {},
            "confidence": "low"
        }
    
def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if value is None or value == '':
            return default
        # Remove currency symbols and commas
        if isinstance(value, str):
            value = value.replace('$', '').replace(',', '').strip()
        return float(value)
    except (ValueError, TypeError):
        return default

def prepare_invoice_chart_data(data, line_items):
    """Prepare chart data for single invoice dashboard"""
    try:
        # Line items breakdown
        line_items_data = {
            'labels': [item.get('Product_Name', f'Item {i+1}')[:30] for i, item in enumerate(line_items[:10])],
            'values': [safe_float(item.get('Line_Total_Amount', 0)) for item in line_items[:10]]
        }
        
        # Cost distribution
        total_amount = safe_float(data.get('Total_Amount', 0))
        amount_excl_gst = safe_float(data.get('Amount_excld_GST', 0))
        gst_amount = safe_float(data.get('GST_Amount', 0))
        freight = safe_float(data.get('Freight_Packing_Handling_chrgs', 0))
        
        if gst_amount == 0 and total_amount > 0 and amount_excl_gst > 0:
            gst_amount = total_amount - amount_excl_gst
        
        cost_distribution = {
            'labels': ['Items Cost', 'GST', 'Freight'],
            'values': [amount_excl_gst - freight, gst_amount, freight]
        }
        
        # Top items by value
        sorted_items = sorted(line_items, key=lambda x: safe_float(x.get('Line_Total_Amount', 0)), reverse=True)[:5]
        top_items = {
            'labels': [item.get('Product_Name', f'Item')[:30] for item in sorted_items],
            'values': [safe_float(item.get('Line_Total_Amount', 0)) for item in sorted_items]
        }
        
        return {
            'line_items_breakdown': line_items_data,
            'cost_distribution': cost_distribution,
            'top_items': top_items,
            'amount_composition': {
                'labels': ['Subtotal', 'GST', 'Freight', 'Total'],
                'values': [amount_excl_gst, gst_amount, freight, total_amount]
            }
        }
    except Exception as e:
        app.logger.error(f"Error preparing chart data: {str(e)}")
        return {}


def process_query_with_oci_llm(user_prompt: str, llm_model: str) -> Dict[str, Any]:
    """Process query with OCI LLM"""
    compartment_id = OCI_CONFIG['compartment_id']
    model_ocid = OCI_CONFIG['model_ocids'][llm_model]
    prompt_content = get_query_prompt(user_prompt)
    
    try:
        if llm_model == "cohere":
            result = cohere_llm_response(compartment_id, current_extracted_text, prompt_content, model_ocid)
            return json.loads(result) if isinstance(result, str) else result
            
        elif llm_model in ["llama-3.1", "llama-3.2", "llama-3.3", "grok-3", "grok-4"]:
            result = meta_llm_response(compartment_id, current_extracted_text, prompt_content, model_ocid)
            return result
            
        else:
            raise ValueError(f"Unsupported LLM model for queries: {llm_model}")
            
    except Exception as e:
        app.logger.error(f"OCI LLM query processing failed: {e}")
        raise

def process_query_with_mistral(user_prompt: str) -> Dict[str, Any]:
    """Process query with Mistral LLM"""
    if not MISTRAL_API_KEY:
        raise ValueError("Mistral API key not configured")
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = get_query_prompt(user_prompt)
    
    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        json_str = result['choices'][0]['message']['content']
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        app.logger.error(f"JSON decode error: {str(e)}")
        return {
            "answer": "Error: Invalid response format from AI model",
            "data_found": False,
            "extracted_fields": {},
            "confidence": "low"
        }
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Request error: {str(e)}")
        raise

def process_with_selected_llm(text: str, llm_model: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Process extracted text using selected LLM"""
    try:
        if OCI_AVAILABLE and llm_model in OCI_CONFIG['model_ocids'] and OCI_CONFIG['model_ocids'][llm_model]:
            return process_with_oci_llm(text, llm_model, file_path)
        else:
            return process_with_mistral_standard(text)
    except Exception as e:
        app.logger.error(f"LLM processing failed: {str(e)}")
        return {"error": f"LLM processing failed: {str(e)}"}

def process_with_oci_llm(text: str, llm_model: str, file_path: Optional[str] = None) -> Dict[str, Any]:
    """Process with OCI LLM based on model type"""
    compartment_id = OCI_CONFIG['compartment_id']
    model_ocid = OCI_CONFIG['model_ocids'][llm_model]
    prompt_content = get_invoice_prompt()
    
    try:
        if llm_model == "cohere":
            result = cohere_llm_response(compartment_id, text, prompt_content, model_ocid)
            return json.loads(result) if isinstance(result, str) else result
            
        elif llm_model == "llama-vision" and file_path:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            result = vision_llm_response(compartment_id, file_bytes, prompt_content, model_ocid)
            return result
            
        elif llm_model in ["llama-3.1", "llama-3.2", "llama-3.3", "grok-3", "grok-4"]:
            result = meta_llm_response(compartment_id, text, prompt_content, model_ocid)
            return result
            
        else:
            raise ValueError(f"Unsupported LLM model: {llm_model}")
            
    except Exception as e:
        app.logger.error(f"OCI LLM processing failed: {e}")
        raise

def process_with_mistral_standard(text: str) -> Dict[str, Any]:
    """Standard invoice processing with Mistral (resilient)"""
    if not MISTRAL_API_KEY:
        return {"error": "Mistral API key not configured"}

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = get_invoice_prompt() + f"\n\n{text[:4000]}"

    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2000
    }

    # use semaphore to limit concurrent LLM calls
    with llm_semaphore:
        try:
            # use tuple timeout: (connect_timeout, read_timeout)
            resp = LLM_SESSION.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=(5, 60))
            resp.raise_for_status()

            # be defensive: validate structure before indexing deeply
            result = resp.json()
            choices = result.get("choices")
            if not choices or not isinstance(choices, list):
                app.logger.error("Mistral returned unexpected response (no choices): %s", repr(result))
                return {"error": "Invalid response from LLM", "raw": result}

            message = choices[0].get("message", {})
            json_str = message.get("content", "")
            if not json_str:
                app.logger.error("Mistral response has empty content: %s", repr(result))
                return {"error": "Empty content from LLM", "raw": result}

            # Extract JSON between markers if present
            if "### START ###" in json_str and "### END ###" in json_str:
                start_idx = json_str.find("### START ###") + len("### START ###")
                end_idx = json_str.find("### END ###")
                json_str = json_str[start_idx:end_idx].strip()

            json_str = json_str.replace("<JSON>", "").replace("</JSON>", "").strip()

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                json_str = fix_common_json_issues(json_str)
                return json.loads(json_str)

        except requests.exceptions.ConnectionError as e:
            app.logger.error("Connection error to Mistral: %s", repr(e))
            # if you want to return a friendly error rather than raising:
            return {"error": "Connection error to LLM", "details": str(e)}
        except requests.exceptions.Timeout as e:
            app.logger.error("Timeout when calling Mistral: %s", repr(e))
            return {"error": "LLM request timeout", "details": str(e)}
        except requests.exceptions.RequestException as e:
            app.logger.error("Request exception to Mistral: %s", traceback.format_exc())
            # re-raise if you want upstream to handle it
            raise

def fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues"""
    # Remove trailing commas
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix unquoted keys (but be careful not to break quoted strings)
    json_str = re.sub(r'(\w+)(\s*):', r'"\1"\2:', json_str)
    
    # Fix single quotes to double quotes (but be careful with contractions)
    json_str = json_str.replace("'", '"')
    
    return json_str

def create_excel_export(batch_results: List[Dict], settings: Dict) -> str:
    """Create Excel file from batch results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"invoice_export_{timestamp}.xlsx"
    excel_path = os.path.join(app.config['EXPORT_FOLDER'], excel_filename)
    
    # Create a simple DataFrame with the batch results
    try:
        flattened_data = []
        for result in batch_results:
            if result.get('status') == 'success' and result.get('data'):
                data = result['data']
                row = {
                    'filename': result.get('filename', 'Unknown'),
                    'document_type': data.get('Document_Type', ''),
                    'abn_number': data.get('ABN_Number', ''),
                    'invoice_date': data.get('Invoice_Date', ''),
                    'invoice_number': data.get('Invoice_Number', ''),
                    'po_number': data.get('PO_Number', ''),
                    'store_id': data.get('Store_ID', ''),
                    'amount_excld_gst': data.get('Amount_excld_GST', ''),
                    'gst_amount': data.get('GST_Amount', ''),
                    'total_amount': data.get('Total_Amount', ''),
                    'processing_time': result.get('processing_time', ''),
                    'status': result.get('status', '')
                }
                flattened_data.append(row)
        
        if flattened_data:
            df = pd.DataFrame(flattened_data)
            df.to_excel(excel_path, index=False, engine='openpyxl')
        else:
            # Create empty Excel file
            df = pd.DataFrame()
            df.to_excel(excel_path, index=False, engine='openpyxl')
            
    except Exception as e:
        app.logger.error(f"Excel creation error: {str(e)}")
        # Create a simple text file as fallback
        with open(excel_path.replace('.xlsx', '.txt'), 'w') as f:
            f.write("Excel export failed. Raw data:\n")
            f.write(json.dumps(batch_results, indent=2))
        excel_path = excel_path.replace('.xlsx', '.txt')
    
    return excel_path

# Cleanup function for old files
def cleanup_old_files():
    """Clean up old temporary files"""
    try:
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['BATCH_FOLDER'], app.config['EXPORT_FOLDER']]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > cleanup_age:
                            try:
                                os.unlink(file_path)
                                app.logger.info(f"Cleaned up old file: {file_path}")
                            except Exception as e:
                                app.logger.warning(f"Failed to cleanup file {file_path}: {e}")
    except Exception as e:
        app.logger.error(f"Cleanup process failed: {e}")

# Schedule cleanup task
import threading

def periodic_cleanup():
    """Run periodic cleanup in background"""
    while True:
        time.sleep(3600)  # Run every hour
        cleanup_old_files()

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    # Ensure all directories exist
    for folder in [app.config['UPLOAD_FOLDER'], app.config['BATCH_FOLDER'], app.config['EXPORT_FOLDER']]:
        os.makedirs(folder, exist_ok=True)
    
    # Log startup information
    app.logger.info("="*50)
    app.logger.info("Enhanced Invoice Processing System with Dashboard Starting")
    app.logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.logger.info(f"Batch folder: {app.config['BATCH_FOLDER']}")
    app.logger.info(f"Export folder: {app.config['EXPORT_FOLDER']}")
    app.logger.info(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)}MB")
    app.logger.info(f"Max batch files: {app.config['MAX_BATCH_FILES']}")
    app.logger.info(f"OCI Available: {OCI_AVAILABLE}")
    app.logger.info(f"Tesseract Available: {bool(tesseract_cmd and os.path.exists(tesseract_cmd))}")
    app.logger.info(f"Mistral API Configured: {bool(MISTRAL_API_KEY)}")
    app.logger.info("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)