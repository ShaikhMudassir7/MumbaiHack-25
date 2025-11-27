from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
import oci
from io import BytesIO
import xlsxwriter

app = Flask(__name__)

# OCI Configuration for AI Insights
def get_oci_client():
    try:
        compartment_id = "ocid1.compartment.oc1..aaaaaaaatompxdjveci7ezznhwukxanejcbqt5omthedknjhmvqdcnsu2gbq"
        CONFIG_PROFILE = "ARPRODMUMBAI"
        config = oci.config.from_file('config', CONFIG_PROFILE)
        endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
        
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config, 
            service_endpoint=endpoint, 
            retry_strategy=oci.retry.NoneRetryStrategy(), 
            timeout=(10, 240)
        )
        return client, compartment_id
    except Exception as e:
        print(f"Error initializing OCI client: {e}")
        return None, None

# Load and preprocess data
def load_and_clean_data():
    try:
        df = pd.read_csv('dummy_jde_data.csv')
        print(f"Loaded dataset with {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        df = pd.DataFrame(columns=[
            'Cost Code', 'Cost Type', 'Account Description', 'Original Budget Amt',
            'Change Order Amt', 'Revised Budget Amt', 'Actual Amount', 
            'Open Commit Amount', 'Estimate At Comp. Amt', 'Budget Var Amount',
            'Company Description'
        ])
    
    df_clean = df.copy()
    
    numeric_columns = ['Original Budget Amt', 'Change Order Amt', 'Revised Budget Amt', 
                      'Actual Amount', 'Open Commit Amount', 'Estimate At Comp. Amt',
                      'Budget Var Amount', 'Actual Units', 'Original Budget Unit',
                      'Change Order Unit', 'Revised Budget Unit', 'Budget Var Units',
                      'Percent Complete']
    
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        else:
            df_clean[col] = 0
    
    text_columns = ['Cost Code', 'Cost Type', 'Account Description', 'Company', 'Company Description']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('').astype(str)
        else:
            df_clean[col] = ''
    
    if 'Company Description' not in df_clean.columns or df_clean['Company Description'].str.strip().eq('').all():
        df_clean['Company Description'] = 'Default Company'
    
    df_clean['Budget_Utilization'] = np.where(
        df_clean['Revised Budget Amt'] != 0,
        (df_clean['Actual Amount'] / df_clean['Revised Budget Amt']) * 100,
        0
    )
    
    def classify_risk(row):
        if row['Revised Budget Amt'] <= 0:
            return 'No Budget'
        budget_util = (row['Actual Amount'] / row['Revised Budget Amt']) * 100
        if budget_util <= 80:
            return 'Low'
        elif budget_util <= 120:
            return 'Medium'
        else:
            return 'Critical'
    
    df_clean['Risk_Factor'] = df_clean.apply(classify_risk, axis=1)
    
    def classify_status(row):
        var_amount = row['Budget Var Amount']
        if var_amount < 0:
            return {'status': 'Over Budget', 'color': 'red'}
        elif var_amount > 0:
            return {'status': 'Under Budget', 'color': 'green'}
        else:
            return {'status': 'On Budget', 'color': 'blue'}
    
    status_result = df_clean.apply(classify_status, axis=1)
    df_clean['Budget_Status'] = status_result.apply(lambda x: x['status'])
    df_clean['Status_Color'] = status_result.apply(lambda x: x['color'])
    
    print(f"Data cleaning completed. Columns: {df_clean.columns.tolist()}")
    return df_clean

df = load_and_clean_data()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/companies')
def get_companies():
    companies = df['Company Description'].unique().tolist()
    return jsonify(companies)

@app.route('/api/cost_codes')
def get_cost_codes():
    company = request.args.get('company', '')
    if company:
        filtered_df = df[df['Company Description'] == company]
    else:
        filtered_df = df
    cost_codes = filtered_df['Cost Code'].unique().tolist()
    return jsonify([str(code) for code in cost_codes if str(code).strip()])

@app.route('/api/account_descriptions')
def get_account_descriptions():
    company = request.args.get('company', '')
    cost_code = request.args.get('cost_code', '')
    filtered_df = df
    if company:
        filtered_df = filtered_df[filtered_df['Company Description'] == company]
    if cost_code:
        filtered_df = filtered_df[filtered_df['Cost Code'] == cost_code]
    accounts = filtered_df['Account Description'].unique().tolist()
    return jsonify([acc for acc in accounts if acc.strip()])

@app.route('/api/kpi_data')
def get_kpi_data():
    company = request.args.get('company', '')
    cost_code = request.args.get('cost_code', '')
    account_desc = request.args.get('account_desc', '')
    
    filtered_df = df
    if company:
        filtered_df = filtered_df[filtered_df['Company Description'] == company]
    if cost_code:
        filtered_df = filtered_df[filtered_df['Cost Code'] == cost_code]
    if account_desc:
        filtered_df = filtered_df[filtered_df['Account Description'] == account_desc]
    
    total_budget = filtered_df['Revised Budget Amt'].sum()
    total_actual = filtered_df['Actual Amount'].sum()
    total_variance = filtered_df['Budget Var Amount'].sum()
    over_running = abs(filtered_df[filtered_df['Budget Var Amount'] < 0]['Budget Var Amount'].sum())
    under_running = filtered_df[filtered_df['Budget Var Amount'] > 0]['Budget Var Amount'].sum()
    risk_counts = filtered_df['Risk_Factor'].value_counts().to_dict()
    
    return jsonify({
        'total_budget': float(total_budget),
        'total_actual': float(total_actual),
        'total_variance': float(total_variance),
        'over_running': float(over_running),
        'under_running': float(under_running),
        'risk_distribution': risk_counts,
        'record_count': len(filtered_df)
    })


@app.route('/api/forecast_data')
def get_forecast_data():
    company = request.args.get('company', '')
    cost_code = request.args.get('cost_code', '')
    account_desc = request.args.get('account_desc', '')
    
    filtered_df = df
    if company:
        filtered_df = filtered_df[filtered_df['Company Description'] == company]
    if cost_code:
        filtered_df = filtered_df[filtered_df['Cost Code'] == cost_code]
    if account_desc:
        filtered_df = filtered_df[filtered_df['Account Description'] == account_desc]
    
    current_budget = filtered_df['Revised Budget Amt'].sum()
    current_actual = filtered_df['Actual Amount'].sum()
    
    if current_budget == 0:
        current_budget = 100000
    if current_actual == 0:
        current_actual = current_budget * 0.8
    
    months = ['Current', 'Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6']
    budget_forecast = [current_budget]
    for i in range(1, 7):
        budget_forecast.append(current_budget * (1 + i * 0.02))
    
    actual_forecast = [current_actual]
    for i in range(1, 7):
        variation = 1 + (i * 0.025) + (random.random() * 0.1 - 0.05)
        actual_forecast.append(current_actual * variation)
    
    return jsonify({
        'months': months,
        'budget_forecast': [float(x) for x in budget_forecast],
        'actual_forecast': [float(x) for x in actual_forecast]
    })

@app.route('/api/drilldown_data')
def get_drilldown_data():
    company = request.args.get('company', '')
    cost_code = request.args.get('cost_code', '')
    account_desc = request.args.get('account_desc', '')
    
    filtered_df = df
    if company:
        filtered_df = filtered_df[filtered_df['Company Description'] == company]
    if cost_code:
        filtered_df = filtered_df[filtered_df['Cost Code'] == cost_code]
    if account_desc:
        filtered_df = filtered_df[filtered_df['Account Description'] == account_desc]
    
    result = filtered_df[['Cost Code', 'Account Description', 'Revised Budget Amt', 
                         'Actual Amount', 'Budget Var Amount', 'Risk_Factor', 'Budget_Status']].to_dict('records')
    
    return jsonify(result)

@app.route('/api/variance_drilldown')
def get_variance_drilldown():
    """Get detailed variance data for drilldown"""
    company = request.args.get('company', '')
    cost_code = request.args.get('cost_code', '')
    account_desc = request.args.get('account_desc', '')
    variance_type = request.args.get('type', 'all')  # 'over', 'under', or 'all'
    
    filtered_df = df
    if company:
        filtered_df = filtered_df[filtered_df['Company Description'] == company]
    if cost_code:
        filtered_df = filtered_df[filtered_df['Cost Code'] == cost_code]
    if account_desc:
        filtered_df = filtered_df[filtered_df['Account Description'] == account_desc]
    
    # Filter based on variance type
    if variance_type == 'over':
        filtered_df = filtered_df[filtered_df['Budget Var Amount'] < 0]
    elif variance_type == 'under':
        filtered_df = filtered_df[filtered_df['Budget Var Amount'] > 0]
    
    # Sort by absolute variance (largest first)
    filtered_df = filtered_df.sort_values('Budget Var Amount', 
                                          key=lambda x: abs(x), 
                                          ascending=False)
    
    result = filtered_df[['Cost Code', 'Account Description', 'Revised Budget Amt', 
                         'Actual Amount', 'Budget Var Amount', 'Risk_Factor', 
                         'Budget_Status', 'Percent Complete']].to_dict('records')
    
    return jsonify(result)

@app.route('/api/account_variance_detail')
def get_account_variance_detail():
    """Get detailed information for a specific account"""
    account = request.args.get('account', '')
    company = request.args.get('company', '')
    cost_code = request.args.get('cost_code', '')
    
    filtered_df = df
    if company:
        filtered_df = filtered_df[filtered_df['Company Description'] == company]
    if cost_code:
        filtered_df = filtered_df[filtered_df['Cost Code'] == cost_code]
    if account:
        filtered_df = filtered_df[filtered_df['Account Description'] == account]
    
    result = filtered_df.to_dict('records')
    
    return jsonify(result)

@app.route('/api/ai_insight', methods=['POST'])
def get_ai_insight():
    try:
        data = request.json
        chart_type = data.get('chart_type', '')
        chart_data = data.get('chart_data', {})
        
        client, compartment_id = get_oci_client()
        
        if not client:
            return jsonify({
                'success': False,
                'insight': 'AI service is currently unavailable. Please check OCI configuration.'
            })
        
        # Create prompt based on chart type
        prompt = create_insight_prompt(chart_type, chart_data)
        
        # Call OCI Generative AI
        content = oci.generative_ai_inference.models.TextContent()
        content.text = prompt
        
        message = oci.generative_ai_inference.models.Message()
        message.role = "USER"
        message.content = [content]
        
        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        chat_request.messages = [message]
        chat_request.max_tokens = 2000
        chat_request.temperature = 0.5
        chat_request.top_p = 0.9
        chat_request.top_k = 0
        
        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya3bsfz4ogiuv3yc7gcnlry7gi3zzx6tnikg6jltqszm2q"
        )
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = compartment_id
        
        chat_response = client.chat(chat_detail)
        
        # Extract the response text
        insight_text = chat_response.data.chat_response.choices[0].message.content[0].text
        
        return jsonify({
            'success': True,
            'insight': insight_text
        })
        
    except Exception as e:
        print(f"Error generating AI insight: {e}")
        return jsonify({
            'success': False,
            'insight': f'Unable to generate AI insight: {str(e)}'
        })

def create_insight_prompt(chart_type, chart_data):
    """Create a contextual prompt for AI based on chart type and data"""
    
    # base_prompt = "You are a financial analyst assistant. Analyze the following budget data and provide 3-4 key insights in a concise, professional manner. Focus on trends, anomalies, and actionable recommendations. Use tabular data in html format whereever needed.\n do the formating in html format for everything (all fonts in white)\n"    
    base_prompt = "You are a financial analyst assistant. Analyze the following budget data and provide 3-4 key insights in a concise, professional manner. Focus on trends, anomalies, and actionable recommendations.\n do the formating in html format for everything (all fonts in white)\n"    
    if chart_type == 'budget_vs_actual':
        prompt = base_prompt + f"""
Chart Type: Budget vs Actual Comparison
Data: {json.dumps(chart_data, indent=2)}

Please analyze:
1. Which accounts show the largest budget variance?
2. Are there patterns of over or under-spending?
3. What are the top concerns and recommendations?
"""
    
    elif chart_type == 'risk_distribution':
        prompt = base_prompt + f"""
Chart Type: Risk Distribution Analysis
Data: {json.dumps(chart_data, indent=2)}

Please analyze:
1. What is the overall risk profile?
2. Which risk categories need immediate attention?
3. What preventive measures should be considered?
"""
    
    elif chart_type == 'variance_analysis':
        prompt = base_prompt + f"""
Chart Type: Variance Analysis
Data: {json.dumps(chart_data, indent=2)}

Please analyze:
1. Which accounts have the most significant variances?
2. Are the variances positive or negative?
3. What corrective actions are recommended?
"""
    
    elif chart_type == 'forecast':
        prompt = base_prompt + f"""
Chart Type: Budget Forecast
Data: {json.dumps(chart_data, indent=2)}

Please analyze:
1. What trends are visible in the forecast?
2. Are we on track to meet budget goals?
3. What risks or opportunities do you see?
"""
    
    else:
        prompt = base_prompt + f"""
Data: {json.dumps(chart_data, indent=2)}

Please provide key insights and recommendations based on this financial data.
"""
    
    return prompt


@app.route('/api/chart_data')
def get_chart_data():
    company = request.args.get('company', '')
    cost_code = request.args.get('cost_code', '')
    account_desc = request.args.get('account_desc', '')
    
    filtered_df = df
    if company:
        filtered_df = filtered_df[filtered_df['Company Description'] == company]
    if cost_code:
        filtered_df = filtered_df[filtered_df['Cost Code'] == cost_code]
    if account_desc:
        filtered_df = filtered_df[filtered_df['Account Description'] == account_desc]
    
    budget_vs_actual_data = filtered_df.groupby('Account Description').agg({
        'Revised Budget Amt': 'sum',
        'Actual Amount': 'sum'
    }).reset_index()
    budget_vs_actual_data = budget_vs_actual_data.nlargest(10, 'Revised Budget Amt')
    
    # Simple risk data for donut chart
    risk_data = filtered_df['Risk_Factor'].value_counts()
    
    # Calculate risk matrix for heatmap
    risk_matrix_data = calculate_risk_matrix(filtered_df)
    
    variance_data = filtered_df.groupby('Account Description').agg({
        'Budget Var Amount': 'sum'
    }).reset_index()
    variance_data['Abs_Variance'] = variance_data['Budget Var Amount'].abs()
    variance_data = variance_data.nlargest(15, 'Abs_Variance')
    
    trend_data = filtered_df.groupby('Account Description').agg({
        'Revised Budget Amt': 'sum',
        'Actual Amount': 'sum',
        'Budget Var Amount': 'sum'
    }).reset_index()
    trend_data = trend_data.nlargest(8, 'Revised Budget Amt')

    return jsonify({
        'budget_vs_actual': {
            'accounts': budget_vs_actual_data['Account Description'].tolist(),
            'budget': budget_vs_actual_data['Revised Budget Amt'].astype(float).tolist(),
            'actual': budget_vs_actual_data['Actual Amount'].astype(float).tolist()
        },
        'risk_distribution': {
            'labels': risk_data.index.tolist(),
            'data': risk_data.values.tolist(),
            'matrix': risk_matrix_data  # Add matrix data
        },
        'variance_analysis': {
            'accounts': variance_data['Account Description'].tolist(),
            'variances': variance_data['Budget Var Amount'].astype(float).tolist()
        },
        'trend_analysis': {
            'accounts': trend_data['Account Description'].tolist(),
            'budget': trend_data['Revised Budget Amt'].astype(float).tolist(),
            'actual': trend_data['Actual Amount'].astype(float).tolist(),
            'variance': trend_data['Budget Var Amount'].astype(float).tolist()
        }
    })

def calculate_risk_matrix(filtered_df):
    """Calculate risk matrix data for heatmap"""
    risk_matrix = {
        'Negligible': {'Improbable': 0, 'Remote': 0, 'Occasional': 0, 'Probable': 0, 'Frequent': 0},
        'Low': {'Improbable': 0, 'Remote': 0, 'Occasional': 0, 'Probable': 0, 'Frequent': 0},
        'Moderate': {'Improbable': 0, 'Remote': 0, 'Occasional': 0, 'Probable': 0, 'Frequent': 0},
        'Significant': {'Improbable': 0, 'Remote': 0, 'Occasional': 0, 'Probable': 0, 'Frequent': 0},
        'Catastrophic': {'Improbable': 0, 'Remote': 0, 'Occasional': 0, 'Probable': 0, 'Frequent': 0}
    }
    
    for _, row in filtered_df.iterrows():
        budget = row['Revised Budget Amt']
        actual = row['Actual Amount']
        
        if budget == 0:
            continue
        
        # Determine Impact based on absolute variance amount
        variance_amt = abs(row['Budget Var Amount'])
        if variance_amt < 10000:
            impact = 'Negligible'
        elif variance_amt < 50000:
            impact = 'Low'
        elif variance_amt < 100000:
            impact = 'Moderate'
        elif variance_amt < 200000:
            impact = 'Significant'
        else:
            impact = 'Catastrophic'
        
        # Determine Likelihood based on budget utilization percentage
        budget_util = (actual / budget * 100) if budget != 0 else 0
        if budget_util < 50:
            likelihood = 'Improbable'
        elif budget_util < 80:
            likelihood = 'Remote'
        elif budget_util < 100:
            likelihood = 'Occasional'
        elif budget_util < 120:
            likelihood = 'Probable'
        else:
            likelihood = 'Frequent'
        
        risk_matrix[impact][likelihood] += 1
    
    return risk_matrix

@app.route('/api/export_chart_data', methods=['POST'])
def export_chart_data():
    try:
        data = request.json
        chart_type = data.get('chart_type', '')
        chart_data = data.get('chart_data', {})
        
        # Create Excel file in memory
        output = BytesIO()
        workbook = xlsxwriter.Workbook(output)
        worksheet = workbook.add_worksheet(chart_type)
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        cell_format = workbook.add_format({'border': 1})
        currency_format = workbook.add_format({'border': 1, 'num_format': '$#,##0'})
        
        # Write data based on chart type
        row = 0
        
        if chart_type == 'budget_vs_actual':
            headers = ['Account', 'Budget', 'Actual', 'Variance']
            for col, header in enumerate(headers):
                worksheet.write(row, col, header, header_format)
            
            row += 1
            accounts = chart_data.get('accounts', [])
            budgets = chart_data.get('budget', [])
            actuals = chart_data.get('actual', [])
            
            for i in range(len(accounts)):
                worksheet.write(row, 0, accounts[i], cell_format)
                worksheet.write(row, 1, budgets[i], currency_format)
                worksheet.write(row, 2, actuals[i], currency_format)
                worksheet.write(row, 3, budgets[i] - actuals[i], currency_format)
                row += 1
                
        elif chart_type == 'risk_distribution':
            headers = ['Risk Level', 'Count']
            for col, header in enumerate(headers):
                worksheet.write(row, col, header, header_format)
            
            row += 1
            labels = chart_data.get('labels', [])
            data_values = chart_data.get('data', [])
            
            for i in range(len(labels)):
                worksheet.write(row, 0, labels[i], cell_format)
                worksheet.write(row, 1, data_values[i], cell_format)
                row += 1
                
        elif chart_type == 'variance_analysis':
            headers = ['Account', 'Variance']
            for col, header in enumerate(headers):
                worksheet.write(row, col, header, header_format)
            
            row += 1
            accounts = chart_data.get('accounts', [])
            variances = chart_data.get('variances', [])
            
            for i in range(len(accounts)):
                worksheet.write(row, 0, accounts[i], cell_format)
                worksheet.write(row, 1, variances[i], currency_format)
                row += 1
                
        elif chart_type == 'forecast':
            headers = ['Month', 'Budget Forecast', 'Actual Forecast']
            for col, header in enumerate(headers):
                worksheet.write(row, col, header, header_format)
            
            row += 1
            months = chart_data.get('months', [])
            budget_forecast = chart_data.get('budget_forecast', [])
            actual_forecast = chart_data.get('actual_forecast', [])
            
            for i in range(len(months)):
                worksheet.write(row, 0, months[i], cell_format)
                worksheet.write(row, 1, budget_forecast[i], currency_format)
                worksheet.write(row, 2, actual_forecast[i], currency_format)
                row += 1
        
        # Auto-fit columns
        for i in range(4):
            worksheet.set_column(i, i, 20)
        
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)