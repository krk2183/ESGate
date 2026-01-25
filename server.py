from google import genai
import pandas as pd
import numpy as np
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, redirect, url_for,app
from flask_cors import CORS
import math, traceback, bcrypt,jwt,datetime,re,requests,json,torch,sqlite3, joblib,gc,random
from google.genai import types
# -------------------------- Flask Setup --------------------------
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = "ENTER_KEY_IF_PRESENT"  
API = 'API_KEY'
if API_KEY=='API-KEY':
    print('Please enter your Gemini API key on Line 13!')
client = genai.Client(api_key=API)
model_n = 'gemini-2.5-flash'

device = torch.device('cpu')  
 
news_prompts =    [
        "Find one recent, important news story about the EU's Corporate Sustainability Reporting Directive (CSRD) scope changes for SMEs.",
        "What is the most recent news regarding the EU's Omnibus package and its effect on ESG compliance for European SMEs?",
        "Detail a recent, major news story on how new EU ESG regulations are altering the supply chain relationship between large European corporations and Azerbaijani SMEs.",
        "Report one recent, significant development concerning the simplification of EU ESG rules and the impact on small and medium-sized enterprises (SMEs) in Europe.",
        "Provide the latest news on Azerbaijan's SME market response to the indirect requirements of the EU's Corporate Sustainability Due Diligence Directive (CSDDD).",
        "Identify a recent, important announcement from the European Commission regarding voluntary ESG reporting standards for SMEs in the EU.",
        "Find one recent, major news story discussing the financial costs or investment trends for SMEs in Europe due to EU ESG compliance.",
        "What is the most recent news on the European Parliament's vote concerning the threshold changes for the CSRD and CSDDD?",
        "Detail a recent, important news story about how banks in Europe are adjusting their lending criteria for SMEs based on EU ESG mandates.",
        "Report one recent, significant development concerning the role of Azerbaijani SMEs in the energy transition and their alignment with EU Green Deal principles.",
        "Provide the latest news on any official EU-Azerbaijan dialogue or working group focused on ESG standards for local businesses.",
        "Find one recent, major news story on the challenges or opportunities for EU-based SMEs in collecting Scope 3 emissions data from non-EU partners like Azerbaijan.",
        "What is the most recent news regarding EFRAG's efforts to simplify the European Sustainability Reporting Standards (ESRS) for smaller companies?",
        "Detail a recent, important news story about the proposed delay in the application deadlines for ESG reporting for listed SMEs in the EU.",
        "Report one recent, significant development concerning the 'greenwashing' concerns among SMEs in Europe linked to new EU regulations.",
        "Provide the latest news on how the EU's Carbon Border Adjustment Mechanism (CBAM) is indirectly affecting Azerbaijani SMEs that supply EU importers.",
        "Find one recent, major news story about government support or subsidy programs for European SMEs to help with new ESG compliance burdens.",
        "What is the most recent news regarding the potential for Azerbaijani SMEs to gain a competitive advantage by proactively adopting ESG principles?",
        "Detail a recent, important news story on the backlash or criticism from business lobbies in Europe concerning the complexity of the CSRD for SMEs.",
        "Report one recent, significant development concerning the role of digitalization and AI tools in helping European SMEs manage their new ESG reporting requirements.",
        "Provide the latest news on whether the CSDDD's due diligence obligations still indirectly apply to non-EU SMEs supplying EU companies despite simplification efforts.",
        "Find one recent, major news story about the establishment of new public-private partnerships to promote ESG readiness among SMEs in Azerbaijan.",
        "What is the most recent news regarding investor interest in European SMEs with high ESG ratings following the introduction of EU regulations?",
        "Detail a recent, important news story on the transfer of ESG technology or knowledge from the EU to partner countries' SMEs, focusing on Azerbaijan.",
        "Report one recent, significant development concerning the impact of the EU Taxonomy Regulation on the financing options for SMEs in Europe.",
        "Provide the latest news on the number of European SMEs expected to be newly excluded from the CSRD's scope due to the proposed Omnibus changes.",
        "Find one recent, major news story about the adoption of a voluntary sustainability reporting standard (VSME) for non-listed SMEs in the EU.",
        "What is the most recent news regarding the legal risks for SMEs in the EU that fail to comply with the new supply chain due diligence requirements?",
        "Detail a recent, important news story on the challenges Azerbaijani SMEs face in proving their 'social' (S) and 'governance' (G) compliance to EU partners.",
        "Report one recent, significant development concerning the harmonization of EU ESG standards with global standards and the implications for European SMEs."
    ]
# -------------------------- DB Setup --------------------------

def init_db():
    user_conn = sqlite3.connect('assets/users.db')
    user_cursor = user_conn.cursor()
    user_cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password BLOB NOT NULL, 
            role TEXT NOT NULL, 
            com_category TEXT 
        )
    ''')
    user_conn.commit()
    user_conn.close()

    pred_conn = sqlite3.connect('assets/database.db')
    pred_cursor = pred_conn.cursor()
    
    pred_cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            company_name TEXT,
            com_category TEXT,  -- <--- NEW COLUMN ADDED HERE
            int_rate REAL,
            default_rate REAL,
            sus_score REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    pred_conn.commit()
    pred_conn.close()

init_db()

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        # Explicitly decode the password column if it's stored as bytes/blob
        if col[0] == 'password' and isinstance(row[idx], str):
            d[col[0]] = row[idx].encode('utf-8')
        else:
            d[col[0]] = row[idx]
    return d



# -------------------------- HelperS --------------------------
from functools import wraps
def get_esg_tier(esg_score):
    if 0< esg_score < 30:
        return "Rank I", "I"
    elif 30 < esg_score < 55:
        return "Rank II", "II"
    elif 55< esg_score < 70:
        return "Rank III", "III"
    elif 70< esg_score < 90:
        return "Rank IV", "IV"
    elif 90< esg_score <100:
        return "Rank V", "V"
    
def get_esg_data(user_id):
    try:
        conn = sqlite3.connect('assets/database.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT int_rate, default_rate, sus_score, created_at
                       FROM predictions
                       WHERE user_id=?
                       ORDER BY created_at DESC
                       LIMIT 2

                       ''',(user_id,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {
                'esg_score':0,
                'int_rate':None,
                'default_rate':None,
                'sus_score':None,
                'esg_improvement':0,
                'has_data':False
            }
        
        # Geting the latest recorded data
        latest  = rows[0]
        int_rate = latest[0]
        default_rate = latest[1]
        sus_score = latest[2]

        if int_rate is not None and default_rate is not None and sus_score is not None:
            current_esg_score = esgatescoref(int_rate,default_rate,sus_score)
        else:
             current_esg_score = 0
        esg_improvement = 0
        if len(rows) >1:
            previous = rows[1]
            prev_int_rate = previous[0]
            prev_def_rate = previous[1]
            prev_sus_rate = previous[2]

            if prev_int_rate is not None and prev_def_rate is not None and prev_sus_rate is not None:
                previous_esg_score = esgatescoref(prev_int_rate,prev_def_rate,prev_sus_rate)
                esg_improvement = current_esg_score - previous_esg_score # Calculating the improvement
        return {
            'esg_score': current_esg_score,
            'int_rate': int_rate,
            'default_rate': default_rate,
            'sus_score': sus_score,
            'esg_improvement': esg_improvement,
            'has_data': True
        }
    except Exception as e:
        print(f"Error fetching ESG data: {e}")
        traceback.print_exc()
        return {
            'esg_score': 0,
            'int_rate': None,
            'default_rate': 0,
            'sus_score': None,
            'esg_improvement': 0,
            'has_data': False
        }


def calculate_progress_percent(esg_score,target_score=80):
    if esg_score >= target_score:
        return 100
    return int((esg_score/target_score)* 100)

# -------------------------- HTML Page Serving --------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            if request.is_json or request.path.startswith('/predict') or request.path.startswith('/history'):
                return jsonify({"message": "Token is missing!"}), 401
            else:
                return redirect(url_for('login_page'))
        
        try:
            # Clean up the token
            if token.startswith('Bearer '):
                token = token.replace('Bearer ', '')
            
            # Decode the token
            data = jwt.decode(
                token, 
                app.config['SECRET_KEY'], 
                algorithms=['HS256']
            )
            
            current_user = data['sub'] 
            role = data['role']
            user_id = data['id']
            
            conn = sqlite3.connect('assets/users.db')
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
            user_record = cursor.fetchone()
            conn.close()

            if user_record is None:
                return jsonify({"message": "User not found."}), 401
            
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Token is invalid."}), 401
        except Exception as e:
            print(f"Token processing error: {e}") 
            return jsonify({"message": f"Token processing error: {str(e)}"}), 401
        
        return f(current_user, role, user_id, *args, **kwargs)
    
    return decorated

@app.route('/')
def home():
    """Serves the landing page (landing.html)."""
    return render_template('landing.html')

@app.route('/login_page')
def login_page():
    return render_template('index.html')

@app.route('/analysis')
@token_required
def analysis(current_user,role,user_id):
    return render_template('user-analysis.html',company_name=current_user.capitalize())


@app.route('/discover_page')
@token_required
def discover_page(current_user,role,user_id):
    return render_template('investor-discover.html')

@app.route('/discover_page_user')
@token_required
def discover_page_user(current_user, role, user_id):
    esg_data = get_esg_data(user_id)    
    esg_tier = 'Satisfactory'

    esg_tier_short = 'IV'
    progress_percentage = calculate_progress_percent(esg_data['esg_score'])
    
    int_rate_display = f"{(esg_data['int_rate'] * 100):.2f}%" if esg_data['int_rate'] is not None else "N/A"
    default_rate_display = f"{(esg_data['default_rate'] * 100):.2f}%" if esg_data['default_rate'] is not None else "N/A"
    sus_score_display = f"{esg_data['sus_score']:.2f}" if esg_data['sus_score'] is not None else "N/A"
    
    return render_template('user-discover.html',
                          username=current_user.capitalize(),
                          esg_score=esg_data['esg_score'],
                          esg_tier=esg_tier,
                          esg_tier_short=esg_tier_short,
                          esg_improvement=abs(esg_data['esg_improvement']),
                          int_rate=int_rate_display,
                          default_rate=default_rate_display,
                          sus_score=sus_score_display,
                          progress_percentage=progress_percentage,
                          has_data=esg_data['has_data'])

def get_user_details(user_id):
    """Fetches user's company category and username for personalized prompts."""
    conn = sqlite3.connect('assets/users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT com_category, username FROM users WHERE id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0], row[1] # category, username
    return "Unspecified", "User"

def generate_roi_prompt(user_data, focus):
    """Constructs the prompt for Gemini based on data availability."""
    company_category, username = get_user_details(user_data['user_id'])
    
    # --- PROMPT FOR COMPANIES WITH EXISTING DATA ---
    if user_data['has_data'] and user_data.get('int_rate') is not None and user_data.get('sus_score') is not None:
        esg_score = user_data['esg_score']
        int_rate_pct = f"{(user_data['int_rate'] * 100):.2f}%"
        default_rate_pct = f"{(user_data['default_rate'] * 100):.2f}%"
        sus_score_out_of_10 = user_data['sus_score']
        
        prompt = f"""
        You are the ESGate AI-Powered ROI Optimizer. Your user is an SME in the '{company_category}' sector in Azerbaijan.
        Current Performance Metrics: Overall ESGate Score: {esg_score:.2f}/100, Interest Rate Probability: {int_rate_pct}, Default Probability: {default_rate_pct}, Sustainability Score (Internal): {sus_score_out_of_10:.2f}/10. Improvement over last period: {user_data['esg_improvement']:.2f} points.
        The user wants a proposal focused on: '{focus}'.
        
        Task: Generate a highly professional, detailed, 3-point **ROI Optimization Proposal** focused on the '{focus}' area, prioritizing the highest financial impact.
        
        For each of the 3 steps, provide a short **Title**, a concise **Description** (1-2 sentences), and a **Predicted_Output_Graph_Value** (a single integer between 10 and 50 representing the predicted % ROI/Compliance gain for that step over 6 months).
        
        Also, generate a general **ROI_Summary**, a **Compliance_Risk_Assessment** for EU market entry, and a **learning_topic** for the Contextual Learning Engine.
        
        Return the response in a strict, single JSON object using the required schema.
        """
        
    # --- PROMPT FOR NEW COMPANIES (NO DATA) ---
    else:
        # No data: General advice
        prompt = f"""
        You are the ESGate AI-Powered ROI Optimizer. Your user is a new SME in the '{company_category}' sector in Azerbaijan, which has no prior ESG data.
        
        Task: Generate a professional, detailed, 3-point **Foundational ROI Proposal** for a company just starting its ESG journey. Focus on low-hanging fruit for the '{company_category}' sector to build a foundation for EU compliance and securing initial financial benefits (Low-Hanging Fruit).
        
        For each of the 3 steps, provide a short **Title**, a concise **Description** (1-2 sentences), and a **Predicted_Output_Graph_Value** (a single integer between 10 and 50 representing the estimated % compliance gain/efficiency improvement for that step over 6 months).
        
        Also, generate a general **ROI_Summary**, a **Compliance_Risk_Assessment** (general assessment), and a **learning_topic** (foundational lesson) for the Contextual Learning Engine.
        
        Return the response in a strict, single JSON object using the required schema.
        """
        
    return prompt


@app.route('/api/generate_roi_plan', methods=['POST'])
@token_required
def generate_roi_plan_endpoint(current_user, role, user_id):
    try:
        data = request.get_json()
        focus = data.get('focus', 'highest_roi')        
        esg_metrics = get_esg_data(user_id)        
        user_category, _ = get_user_details(user_id)        
        esg_metrics['user_id'] = user_id
        esg_metrics['user_category'] = user_category
        gemini_prompt = generate_roi_prompt(esg_metrics, focus)
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "has_data": {"type": "boolean"},
                    "roi_summary": {"type": "string"},
                    "compliance_risk_assessment": {"type": "string"},
                    "learning_topic": {"type": "string"},
                    "proposals": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "predicted_output_graph_value": {"type": "integer"}
                            },
                            "required": ["title", "description", "predicted_output_graph_value"]
                        }
                    }
                },
                "required": ["has_data", "roi_summary", "compliance_risk_assessment", "learning_topic", "proposals"]
            }
        )
        
        # Call Gemini
        response = client.models.generate_content(
            model=model_n, 
            contents=gemini_prompt,
            config=config
        )
        
        ai_output = json.loads(response.text)
        return jsonify(ai_output), 200

    except Exception as e:
        print(f"Error in Gemini ROI plan generation: {e}")
        traceback.print_exc()
        
        # Fallback response (Crucial for MVP demonstration)
        fallback_response = {
            "has_data": False,
            "roi_summary": "AI Service unavailable. Default foundational advice provided.",
            "compliance_risk_assessment": "Immediate high risk due to lack of documented compliance efforts.",
            "learning_topic": "Foundational ESG for SMEs",
            "proposals": [
                {
                    "title": "Establish Digital Record Keeping",
                    "description": "Start migrating all financial and operational records to a structured digital format for audit readiness.",
                    "predicted_output_graph_value": 25
                },
                {
                    "title": "Define Core Metrics (Energy/Waste)",
                    "description": "Select three key environmental metrics (e.g., monthly electricity consumption, waste volume) and start tracking them manually.",
                    "predicted_output_graph_value": 30
                },
                {
                    "title": "Formalize Management Structure",
                    "description": "Document the basic management team, ownership, and formal policies (e.g., anti-corruption) required for basic governance standards.",
                    "predicted_output_graph_value": 20
                }
            ]
        }
        return jsonify(fallback_response), 500

@app.route('/roi_imp') 
# @jwt_required() 
def roi_imp():
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        username_for_template = "User" 
    else:
        return redirect(url_for('login_page'))
    
    return render_template('roi-img.html', 
        username=username_for_template) 


@app.route('/sign-up-page')         
def signup_page():
    """Serves the sign-up page."""
    return render_template('sign-up.html')

@app.route('/user_page')
@token_required
def user_page(current_user,role,user_id):
    if role == 'enterprise':
        esg_data = get_esg_data(user_id)
        
        score = esg_data.get('esg_score')
        esg_tier = "Satisfactory "
        esg_tier_short = "IV"
        
        progress_percentage = calculate_progress_percent(esg_data['esg_score'])
        
        esg_improvement_positive = esg_data['esg_improvement'] > 0
        
        environmental_score = 68
        social_score = 79
        governance_score = 82
        
        return render_template('home-page.html', 
                            company_name=current_user.capitalize(),
                            last_esg_score=esg_data['esg_score'],
                            esg_tier=esg_tier,
                            esg_tier_short=esg_tier_short,
                            esg_improvement=abs(esg_data['esg_improvement']),
                            esg_improvement_positive=esg_improvement_positive,
                            environmental_score=environmental_score,
                            social_score=social_score,
                            governance_score=governance_score,
                            progress_percentage=progress_percentage,
                            has_data=esg_data['has_data'],
                            com_category = 'Textile')
    elif role=='investor':
        return render_template('investor-home.html')
    else:
        raise ValueError('Role not found')
    

@app.route('/admin-page')
@token_required
def admin_page(current_user, role,user_id):
    """Serves the admin dashboard."""
    # You might want to add @admin_required here later for more security
    if role != 'admin':
        return jsonify({"message": "Access forbidden: Admins only"}), 403
    return render_template('admin-page.html')

@app.route('/specialist-page')
@token_required
def specialist_page(current_user, role,user_id):
    """Serves the specialist dashboard."""
    if role not in ['specialist', 'admin']:
        return jsonify({"message": "Access forbidden"}), 403
    return render_template('specialist-page.html')
@app.route('/settings_page')
def settings_page():
    return render_template('investor-settings.html')


# -------------------------- Score History & Comparison --------------------------
def get_history(comp_name,role,user_id):
    conn = sqlite3.connect('assets/database.db') 
    cursor = conn.cursor()
    cursor.execute('SELECT int_rate, default_rate, sus_score FROM users WHERE username = ?', (comp_name,))
    row = cursor.fetchone()
    int_rate,default_rate,sus_score = row
    return int_rate,default_rate,sus_score

def calculate_percentage_change(old_value, new_value):
    if old_value is None or new_value is None or not isinstance(old_value, (int, float)) or not isinstance(new_value, (int, float)):
        return None
    if old_value == 0:
        # Handle division by zero - infinite change or undefined
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0.0 
    try:
        return ((new_value - old_value) / abs(old_value)) * 100
    except Exception:
        return None # General catch-all

def build_history_analysis_prompt(company_name, history_data, improvements):
    
    first_pred = history_data[0]
    last_pred = history_data[-1]

    int_rate_change_str = f"{improvements.get('int_rate_change', 'N/A'):.1f}%" if improvements.get('int_rate_change') is not None else "N/A"
    default_rate_change_str = f"{improvements.get('default_rate_change', 'N/A'):.1f}%" if improvements.get('default_rate_change') is not None else "N/A"
    sus_score_change_str = f"{improvements.get('sus_score_change', 'N/A'):.1f}%" if improvements.get('sus_score_change') is not None else "N/A"

    prompt = f"""
    You are a financial and ESG analyst AI. Analyze the historical performance trends for the company '{company_name}'.

    Historical Data Overview:
    - Number of data points: {len(history_data)}
    - First Prediction Date: {first_pred['created_at']}
    - Last Prediction Date: {last_pred['created_at']}

    Metric Changes (from first to last prediction):
    - Interest Rate: 
        - Started at: {f"{(first_pred['int_rate'] * 100):.2f}%" if first_pred['int_rate'] is not None else 'N/A'}
        - Ended at: {f"{(last_pred['int_rate'] * 100):.2f}%" if last_pred['int_rate'] is not None else 'N/A'}
        - Change: {int_rate_change_str} (Note: A decrease is an improvement)
    - Default Probability:
        - Started at: {f"{(first_pred['default_rate'] * 100):.2f}%" if first_pred['default_rate'] is not None else 'N/A'}
        - Ended at: {f"{(last_pred['default_rate'] * 100):.2f}%" if last_pred['default_rate'] is not None else 'N/A'}
        - Change: {default_rate_change_str} (Note: A decrease is an improvement)
    - Sustainability Score (/10):
        - Started at: {f"{first_pred['sus_score']:.2f}" if first_pred['sus_score'] is not None else 'N/A'}
        - Ended at: {f"{last_pred['sus_score']:.2f}" if last_pred['sus_score'] is not None else 'N/A'}
        - Change: {sus_score_change_str} (Note: An increase is an improvement)

    Task:
    Provide a brief analysis commenting on the overall trend for each metric based on the percentage change provided. 
    - Is the change significant? 
    - Does it indicate improvement or decline in that area? 
    - Keep the comments concise (1-2 sentences per metric).

    Return the result in strict JSON format:
    {{
      "interest_rate_comment": "...",
      "default_rate_comment": "...",
      "sus_score_comment": "...",
      "overall_summary": "..." 
    }}
    """
    return prompt

def get_mistral_history_analysis(company_name, history_data, improvements):
    """ Calls Mistral API to get analysis comments. """
    payload = {
        "model": "mistralai/mistral-small-3.2-24b-instruct:free", 
        "messages": [
            {"role": "user", "content": build_history_analysis_prompt(company_name, history_data, improvements)}
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"}, 
        "max_tokens": 400 
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload)
        response.raise_for_status() 
        result = response.json()
        return parse_mistral_api_output(result)
    except requests.exceptions.RequestException as e:
         print(f"Mistral API request failed: {e}")
         return {"error": f"AI analysis request failed: {str(e)}"}
    except Exception as e:
        print(f"Error processing Mistral response: {e}")
        return {"error": "Failed to process AI analysis response."}


@app.route('/history', methods=['POST'])
@token_required
def history(current_user, role,user_id):
    company_name = current_user
    if not company_name:
        return jsonify({"error": "Company name was not provided - SERVER-SIDE ERROR."}), 400

    company_name_lower = str(company_name).lower().strip()
    history_data = []
    improvements = {}
    ai_analysis = {}

    try:
        # user_conn = sqlite3.connect('assets/users.db')
        # user_cursor = user_conn.cursor()
        # user_cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
        # user_row = user_cursor.fetchone()
        # user_conn.close()

        # if user_row is None:
        #     return jsonify({"error": "User not found in user database."}), 404
        # user_id = user_row[0]

        pred_conn = sqlite3.connect('assets/database.db')
        pred_conn.row_factory = sqlite3.Row
        pred_cursor = pred_conn.cursor()

        print(f"--- History Check ---")
        print(f"Current User (from token): {current_user}")
        print(f"Target User ID: {user_id}")
        print(f"Target Company Name (Lowercase for Query): '{company_name_lower}'")
        print(f"SQL Query: SELECT int_rate, default_rate, sus_score, created_at FROM predictions WHERE user_id = ? AND lower(company_name) = ? ORDER BY created_at ASC")
        print(f"SQL Params: ({user_id}, '{company_name_lower}')")

        pred_cursor.execute('''
            SELECT int_rate, default_rate, sus_score, created_at
            FROM predictions
            WHERE user_id = ? AND lower(company_name) = ?
            ORDER BY created_at ASC
        ''', (user_id, company_name_lower)) 

        rows = pred_cursor.fetchall()
        print(f"Rows found by query: {len(rows)}")
        pred_conn.close()

        if not rows:
            print(f"No history found for user_id={user_id}, company_name='{company_name_lower}'") 
            return jsonify({"message": "No history found for this company."}), 200

        history_data = [dict(row) for row in rows]


        if len(history_data) >= 2:
            first_int_rate = next((p['int_rate'] for p in history_data if p['int_rate'] is not None), None)
            last_int_rate = next((p['int_rate'] for p in reversed(history_data) if p['int_rate'] is not None), None)

            first_default_rate = next((p['default_rate'] for p in history_data if p['default_rate'] is not None), None)
            last_default_rate = next((p['default_rate'] for p in reversed(history_data) if p['default_rate'] is not None), None)

            first_sus_score = next((p['sus_score'] for p in history_data if p['sus_score'] is not None), None)
            last_sus_score = next((p['sus_score'] for p in reversed(history_data) if p['sus_score'] is not None), None)

            improvements['int_rate_change'] = calculate_percentage_change(first_int_rate, last_int_rate)
            improvements['default_rate_change'] = calculate_percentage_change(first_default_rate, last_default_rate)
            improvements['sus_score_change'] = calculate_percentage_change(first_sus_score, last_sus_score)
            ai_analysis = get_mistral_history_analysis(company_name, history_data, improvements) 
        else:
            improvements['message'] = "Need at least two data points to calculate change."
            ai_analysis['message'] = "Need at least two data points for AI analysis."

    except sqlite3.Error as e:
        print(f"Database error fetching history: {e}")
        return jsonify({"error": f"Database error fetching history: {str(e)}"}), 500
    except Exception as e:
        print(f"Error processing history: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    gc.collect()

    return jsonify({
        "company_name": company_name, 
        "history": history_data,
        "improvement_metrics": improvements,
        "ai_analysis": ai_analysis
    }), 200

# -------------------------- Login & Sign-up --------------------------
@app.route('/sign-up', methods=['POST'])
def signup_submit():
    username = request.json.get('username')
    password = request.json.get('password')
    role = request.json.get('role', 'user')
    category = request.json.get('category','unspecified')
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    conn = sqlite3.connect('assets/users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return jsonify({"message": "Username already exists"}), 400
    cursor.execute("INSERT INTO users (username, password, role,com_category) VALUES (?, ?, ?, ?)", (username, hashed_password, role,category))
    conn.commit()
    conn.close()
    return jsonify({"message": "Sign-up successful"}), 200

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json(force=True)
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return jsonify({"message": "Request body must be valid JSON."}), 400
        
    username = data.get('username')
    password = data.get('password')
    remember_me = data.get('remember')

    if not username or not password:
        return jsonify({"message": "Missing username or password in request."}), 400


    conn = sqlite3.connect('assets/users.db') 
    cursor = conn.cursor()
    
    cursor.execute("SELECT password, role, id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        stored_hash = result[0]
        
        input_password_bytes = password.encode('utf-8')

        if bcrypt.checkpw(input_password_bytes, stored_hash):
            role = result[1]
            user_id = result[2]
            if remember_me:
                token_span = datetime.timedelta(days=15)
            else:
                token_span = datetime.timedelta(minutes=30)


            
            payload = {
                'sub': username,
                'role': role,
                'id': user_id,
                'iat': datetime.datetime.now(datetime.UTC),
                'exp': datetime.datetime.now(datetime.UTC) + token_span
            }
            token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
            
            return jsonify({"token": token, "role": role}), 200

    return jsonify({"message": "Invalid credentials"}), 401

class NeuralNet(nn.Module):
    """Adaptable PyTorch Neural Network Model for Classification or Regression that can be reused in the future. <--- !!!!!! """
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.2, is_classification=False):
        super(NeuralNet, self).__init__()
        self.is_classification = is_classification
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout)
            ])
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        out = torch.sigmoid(self.out(x))
        return out

# -------------------------- Prediction Models Setup --------------------------
scaler = joblib.load('assets/scaler.pkl')
model_reg_loaded = NeuralNet(input_size=25, hidden_sizes=[256,128,64], output_size=1).to(device)
model_reg_loaded.load_state_dict(torch.load("assets/interest_rate_prediction_model_best.pth", map_location=device))

# Residual MLP for default prediction
model = ResidualMLP(48).to(device)
model.load_state_dict(torch.load("assets/residual_mlp_sme.pth", map_location=device))
preprocessor = joblib.load("assets/preprocessor_sme_advanced.joblib")

income_bins = [4000.0, 35000.0, 49000.0, 63000.0, 86000.0, 6000000.0]
derived_feature_info = {
     'management_team_experience_median': 4.0,
     'income_bins': income_bins,
}
feature_names_order_reg = ['annual_revenue',
                           'office_ownership_status',
                           'management_team_experience',
                           'loan_amount',
                           'loan_percent_revenue',
                           'default_history',
                           'credit_history_length',
                           'years_in_operation_sqrt',
                           'debt_to_revenue_ratio',
                           'loan_to_operating_years_ratio_sqrt',
                           'loan_to_credit_hist_ratio',
                           'management_turnover_fraction_sqrt',
                           'log_revenue',
                           'log_loan_amount',
                           'op_years_sqrt_x_team_exp',
                           'company_size_group_Startup (0-5)',
                           'company_size_group_Growth (6-15)',
                           'company_size_group_Mature (16-30)',
                           'company_size_group_Established (31-50)',
                           'company_size_group_Legacy (51+)',
                           'revenue_bracket_1',
                           'revenue_bracket_2',
                           'revenue_bracket_3',
                           'revenue_bracket_4',
                           'revenue_bracket_5']


numerical_cols_to_scale_fit = ['annual_revenue',
                           'office_ownership_status',
                           'management_team_experience',
                           'loan_amount',
                           'loan_percent_revenue',
                           'default_history',
                           'credit_history_length',
                           'repayment_status',
                           'years_in_operation_sqrt',
                           'debt_to_revenue_ratio',
                           'loan_to_operating_years_ratio_sqrt',
                           'loan_to_credit_hist_ratio',
                           'management_turnover_fraction_sqrt',
                           'log_revenue',
                           'log_loan_amount',
                           'op_years_sqrt_x_team_exp'
                           ]

# --- Mistral AI setup ---
MISTRAL_API_KEY = 'API-KEY'
if MISTRAL_API_KEY=='API-KEY':
    print('Please enter your API key on Line 847!')
 
MISTRAL_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

# -------------------------- Helper Functions --------------------------
def save_prediction_metric(user_id, company_name, category, metric_name, metric_value):
    """
    Saves a single prediction metric by updating a recent record or inserting a new one.
    """
    print(f"\n--- Saving Metric ---")
    print(f"User ID: {user_id}")
    print(f"Metric: {metric_name}, Value: {metric_value}")

    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    company_name_lower = str(company_name).lower().strip() if company_name else ''

    if metric_name not in ['int_rate', 'default_rate', 'sus_score']:
        conn.close()
        raise ValueError("Invalid metric_name provided.")

    try: 
        # Check for recent record
        cursor.execute("""
            SELECT id FROM predictions
            WHERE user_id = ? AND lower(company_name) = ? AND created_at >= datetime('now', '-5 minutes')
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id, company_name_lower)) 
        recent_record = cursor.fetchone()

        if recent_record:
            record_id = recent_record[0]
            print(f"Updating record {record_id}...")
            query = f"UPDATE predictions SET {metric_name} = ? WHERE id = ?"
            cursor.execute(query, (metric_value, record_id))
        else:
            print(f"Inserting new record...")
            # INSERT added for compatibility
            query = f"INSERT INTO predictions (user_id, company_name, com_category, {metric_name}) VALUES (?, ?, ?, ?)"
            cursor.execute(query, (user_id, company_name_lower, category, metric_value))

        conn.commit()
        print(f"Saved successfully.")

    except sqlite3.Error as e:
        print(f"DATABASE ERROR during save: {e}")
        conn.rollback()
    finally:
        conn.close()
        gc.collect()

# Quality of life improvement
def is_finite(num):
    try:
        return (num is not None) and np.isfinite(float(num))
    except Exception:
        return False
    
def validate_prediction_value(val,val_name='prediction',min_value=None,max_val=None):
    if val is None:
        return False, f"{val_name} is None"
    if not is_finite(val):
        return False, f"{val_name} is not finite: {val}"
    v = float(val)
    if (min_value is not None and v < min_value) or (max_val is not None and v > max_val):
        return False, f"{val_name} out of expected range [{min_value}, {max_val}]: {v}"
    return True, None

# predict_company_metrics, predict_default_sme, calculate_sustainability_score
def predict_company_metrics(model_to_use, scaler, derived_feature_info, feature_names_order, numerical_cols_to_scale_fit, device, metric_type,
                            years_in_operation, annual_revenue, office_ownership_status, management_team_experience, loan_amount,
                            default_history, credit_history_length, repayment_status):

    feature_values = {
        'years_in_operation': years_in_operation,
        'annual_revenue': annual_revenue,
        'office_ownership_status': office_ownership_status,
        'management_team_experience': management_team_experience,
        'loan_amount': loan_amount,
        'default_history': default_history,
        'credit_history_length': credit_history_length,
        'repayment_status': repayment_status  # Needed for scaling
    }
    user_df = pd.DataFrame([feature_values])

    # Derived features
    user_df['years_in_operation_sqrt'] = np.sqrt(user_df['years_in_operation'])
    years_in_op_sqrt = user_df['years_in_operation_sqrt'].iloc[0]
    user_df['debt_to_revenue_ratio'] = np.where(user_df['annual_revenue'] > 0, user_df['loan_amount'] / user_df['annual_revenue'], 0)
    user_df['loan_to_operating_years_ratio_sqrt'] = np.where(user_df['years_in_operation_sqrt'] > 0, user_df['loan_amount'] / user_df['years_in_operation_sqrt'], 0)
    user_df['loan_to_credit_hist_ratio'] = np.where(user_df['credit_history_length'] > 0, user_df['loan_amount'] / user_df['credit_history_length'], 0)
    user_df['management_turnover_fraction_sqrt'] = np.where(years_in_op_sqrt > 0, user_df['management_team_experience'] / years_in_op_sqrt, 0)
    user_df['log_revenue'] = np.log1p(user_df['annual_revenue'])
    user_df['log_loan_amount'] = np.log1p(user_df['loan_amount'])
    emp_length_imputed_user = user_df['management_team_experience'].fillna(derived_feature_info.get('management_team_experience_median', 0))
    user_df['op_years_sqrt_x_team_exp'] = user_df['years_in_operation_sqrt'] * emp_length_imputed_user

    # Add loan_percent_revenue which is also in the feature list
    user_df['loan_percent_revenue'] = np.where(user_df['annual_revenue'] > 0, 
                                               (user_df['loan_amount'] / user_df['annual_revenue']) * 100, 
                                               0)

    # Categoricals
    company_age_bins = [0, 5, 15, 30, 50, np.inf]
    company_age_labels = ['Startup (0-5)', 'Growth (6-15)', 'Mature (16-30)', 'Established (31-50)', 'Legacy (51+)']
    user_df['company_size_group'] = pd.cut(user_df['years_in_operation'], bins=company_age_bins, labels=company_age_labels, right=True, include_lowest=True)
    user_df['revenue_bracket'] = pd.cut(user_df['annual_revenue'], bins=derived_feature_info['income_bins'], labels=[1, 2, 3, 4, 5], include_lowest=True)
    user_df = pd.get_dummies(user_df, columns=['company_size_group'], prefix='company_size_group')
    user_df['revenue_bracket'] = user_df['revenue_bracket'].astype('category')
    user_df = pd.get_dummies(user_df, columns=['revenue_bracket'], prefix='revenue_bracket')

    # Drop years_in_operation AFTER creating all derived features
    user_df.drop('years_in_operation', axis=1, inplace=True)
    
    # First, ensure all numerical columns exist
    for col in numerical_cols_to_scale_fit:
        if col not in user_df.columns:
            user_df[col] = 0.0
    
    # Scale only the numerical columns (this includes repayment_status)
    scaled_values = scaler.transform(user_df[numerical_cols_to_scale_fit])
    user_df[numerical_cols_to_scale_fit] = scaled_values   
    # NOW drop repayment_status since it's not in the final feature set
    user_df.drop('repayment_status', axis=1, inplace=True, errors='ignore')
    
    # Reindex to match expected feature order (without repayment_status)
    user_df = user_df.reindex(columns=feature_names_order, fill_value=0)
    user_df = user_df.fillna(0).astype(np.float32)
    
    # Convert to tensor for model input
    user_input_tensor = torch.tensor(user_df.values, dtype=torch.float32).to(device)

    model_to_use.eval()
    with torch.no_grad():
        output = model_to_use(user_input_tensor)
        if metric_type == 'rate':
            prediction = output/100.0
        elif metric_type == 'default':
            prediction = torch.sigmoid(output)
        else:
            raise ValueError("Invalid metric_type specified. Use 'rate' or 'default'.")
    gc.collect()
    torch.cuda.empty_cache()
    return prediction.item()


def build_prompt(metrics_dict):
    int_rate_percent = f"{metrics_dict.get('int_rate', 0) * 100:.2f}%" if metrics_dict.get('int_rate') is not None else "N/A"
    default_rate_percent = f"{metrics_dict.get('default_rate', 0) * 100:.2f}%" if metrics_dict.get('default_rate') is not None else "N/A"

    return f"""
        You are a financial and ESG advisor AI. A company has the following metrics:

        Overall ESGate Score: {metrics_dict.get('esgatescore')} out of 100
        Interest Rate: {int_rate_percent}
        Default Probability: {default_rate_percent}
        Sustainability Score: {metrics_dict.get('sus_score', 'N/A')} out of 10
        Additional Notes: {metrics_dict.get('notes', '')}

        Task:
        1. Provide a concise summary (2-3 sentences) of the company's financial and sustainability health.
        2. In your summary, you MUST classify the company's risk based on its Default Probability using these exact categories:
           - Low Risk: Below 30%
           - Average Risk: 30% to 45%
           - High Risk: Above 50%
        3. Highlight strengths and weaknesses in separate bullet points.
        4. Suggest 2-3 actionable recommendations to improve financial or sustainability performance.

        Return the result in strict JSON format:
        {{
        "summary": "...",
        "strengths": ["..."],
        "weaknesses": ["..."],
        "recommendations": ["..."]
        }}
        """
# def parse_mistral_output(response_json):
#     try:
#         output_text = response_json["choices"][0]["text"]
#         # Remove REMOVES json code is present 
#         match = re.search(r"```json\s*(\{.*\})\s*```", output_text, re.DOTALL)
#         if match:
#             output_text = match.group(1)
#         return json.loads(output_text)
#     except Exception as e:
#         return {"error": f"Failed to parse Mistral output: {str(e)}", "raw_text": response_json}

def predict_default_sme(sample_dict):
    df = pd.DataFrame([sample_dict])

    # --- Feature engineering ---
    massive_features = ["Annual Income", "Maximum Open Credit", "Current Loan Amount",
                        "Current Credit Balance", "Monthly Debt"]
    for col in massive_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    df["Debt_to_Income_Ratio"] = df["Monthly Debt"] / (df["Annual Income"] + 1e-6)
    df["Credit_Utilization"] = df["Current Credit Balance"] / (df["Maximum Open Credit"] + 1e-6)
    df["Loan_to_Income_Ratio"] = df["Current Loan Amount"] / (df["Annual Income"] + 1e-6)
    df["Credit_Problem_Score"] = (
        df["Number of Credit Problems"] +
        df["Bankruptcies"] * 2 +
        (df["Tax Liens"] > 0).astype(int)
    )

    # --- Ensure all columns from training exist ---
    numeric_cols_fitted = preprocessor.transformers_[0][2]
    categorical_cols_fitted = preprocessor.transformers_[1][2]

    for col in numeric_cols_fitted + categorical_cols_fitted:
        if col not in df.columns:
            df[col] = 0 if col in numeric_cols_fitted else "Unknown"

    expected_feature_order = [
        'num__Annual Income', 'num__Tax Liens', 'num__Number of Open Accounts',
        'num__Years of Credit History', 'num__Maximum Open Credit',
        'num__Number of Credit Problems', 'num__Months since last delinquent',
        'num__Bankruptcies', 'num__Current Loan Amount',
        'num__Current Credit Balance', 'num__Monthly Debt', 'num__Credit Score',
        'num__Debt_to_Income_Ratio', 'num__Credit_Utilization',
        'num__Loan_to_Income_Ratio', 'num__Credit_Problem_Score',
        'cat__Home Ownership_Have Mortgage', 'cat__Home Ownership_Home Mortgage',
        'cat__Home Ownership_Own Home', 'cat__Home Ownership_Rent',
        'cat__Years in current job_1 year', 'cat__Years in current job_10+ years',
        'cat__Years in current job_2 years', 'cat__Years in current job_3 years',
        'cat__Years in current job_4 years', 'cat__Years in current job_5 years',
        'cat__Years in current job_6 years', 'cat__Years in current job_7 years',
        'cat__Years in current job_8 years', 'cat__Years in current job_9 years',
        'cat__Years in current job_< 1 year', 'cat__Purpose_business loan',
        'cat__Purpose_buy a car', 'cat__Purpose_buy house',
        'cat__Purpose_debt consolidation', 'cat__Purpose_educational expenses',
        'cat__Purpose_home improvements', 'cat__Purpose_major purchase',
        'cat__Purpose_medical bills', 'cat__Purpose_moving', 'cat__Purpose_other',
        'cat__Purpose_renewable energy', 'cat__Purpose_small business',
        'cat__Purpose_take a trip', 'cat__Purpose_vacation',
        'cat__Purpose_wedding', 'cat__Term_Long Term', 'cat__Term_Short Term'
    ]

    # Transform input
    X_input = preprocessor.transform(df)

    #  Ensure correct shape
    X_df = pd.DataFrame(X_input, columns=preprocessor.get_feature_names_out())
    for col in expected_feature_order:
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df.reindex(columns=expected_feature_order, fill_value=0)

    # Convert to tensor
    X_input = torch.tensor(X_df.values, dtype=torch.float32).to(device)

    # ------ Load model and predict -------
    model_def = ResidualMLP(X_input.shape[1]).to(device)
    model_def.load_state_dict(torch.load("assets/residual_mlp_sme.pth", map_location=device))
    model_def.eval()

    with torch.no_grad():
        pred = model_def(X_input).cpu().numpy()[0][0]
    gc.collect()
    torch.cuda.empty_cache()
    return float(pred)


def calculate_sustainability_score(company_metrics, sector_averages, weights=None, tolerance=0.2, max_penalty=1.0):
    if weights is None:
        weights = {'energy_efficiency': 0.4, 'carbon_intensity': 0.3, 'water_usage': 0.3}

    normalized = {}
    for key in company_metrics:
        normalized[key] = company_metrics[key] / sector_averages[key]

    prelim_score = sum(weights[key] * normalized[key] for key in normalized) * 10 / sum(weights.values())

    penalty = 0
    for key in normalized:
        deviation = abs(normalized[key] - 1)  
        if deviation > tolerance:
            penalty += min(max_penalty, (deviation - tolerance) * max_penalty / (1 - tolerance))

        sustainability_score = max(0, prelim_score - penalty) 
    gc.collect()
    torch.cuda.empty_cache()
    return [round(sustainability_score, 2),round(penalty,2)]

def get_mistral_summary(metrics_dict):
    payload = {
        "model": "mistralai/mistral-small-3.2-24b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are a financial and ESG advisor AI."},
            {"role": "user", "content": build_prompt(metrics_dict)}
        ],
        "temperature": 0.7,
        'response_format':{'type':'json_object'},
        "max_tokens": 500
    }
    try:
        response = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload)
        if response.status_code != 200:
            MISTRAL_API_KEY_S = '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'
            HEADERS_S = {
                "Authorization": f"Bearer {MISTRAL_API_KEY_S}",
                "Content-Type": "application/json"
            }
            print("Primary API key failed, trying backup key...")
            response = requests.post(MISTRAL_API_URL, headers=HEADERS_S, json=payload)

        response.raise_for_status()
        result = response.json()
        print(f'Company Summary API Response: {result}')
    
            # Using the second API_KEY
        if response.status_code != 200:
            MISTRAL_API_KEY_S = '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'
            HEADERS_S = {
                "Authorization": f"Bearer {MISTRAL_API_KEY_S}",
                "Content-Type": "application/json"
            }
            response = requests.post(MISTRAL_API_URL, headers=HEADERS_S, json=payload)
            # return {"error": f"Mistral API error: {response.text}"}
            result = response.json()

        return parse_mistral_api_output(result)
    except requests.exceptions.RequestException as e:
        print(f"Mistral API Request Failed for comapany summary: {e}")
        return {'error': f'AI Analysis request failed: {str(e)}'}
    except Exception as e:
        print('Error processing company summary response: ',e)
        traceback.print_exc()
        return {'error': f'Failed to process AI Analysis reponse: {e}'}
    
@app.route('/get_data',methods=['GET']) 
@token_required 
def db_to_csv(current_user,role,user_id):
    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        conn.close()
        return jsonify({"error": "Invalid User ID format."}), 400
    cursor.execute(
        'SELECT int_rate, default_rate, sus_score FROM predictions WHERE user_id=?', 
        (user_id_int,) 
    )
    data = cursor.fetchall()
    conn.close()
    df = pd.DataFrame(
        data, 
        columns=['interest_rate', 'default_rate', 'sus_score']
    )    
    df = df.apply(pd.to_numeric, errors='coerce')
    csv_string = df.to_csv(index=False)
    
    response = app.make_response(csv_string)
    response.headers["Content-Disposition"] = "attachment; filename=predictions_export.csv"
    response.headers["Content-type"] = "text/csv"
    
    return response

# -------------------------- Prediction Endpoints --------------------------

@app.route('/company_count',methods=['POST','GET'])
@token_required
def company_count(current_user,role,user_id):
    print('Attemping to get METRICS_PANE information...')
    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT COUNT(id) FROM predictions
        ''')
    count = cursor.fetchall()
    return jsonify({'num_companies':count})




@app.route('/avg_esg', methods=['POST','GET'])
@token_required
def avg_esg(current_user, role, user_id):
    conn = sqlite3.connect('assets/database.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT AVG(int_rate) AS avg_int_rate, AVG(default_rate) AS avg_def_rate, AVG(sus_score) AS avg_sus_score
    FROM predictions;               
    ''')
    averages = cursor.fetchone()
    conn.close()

    esg_score = 0 

    if averages and averages[0] is not None:
        avg_int_rate, avg_def_rate, avg_sus_score = averages
        try:
            esg_score = esgatescoref(avg_int_rate, avg_def_rate, avg_sus_score)
        except Exception:
            esg_score = 0
            
    return jsonify({'esg_score': esg_score})



@app.route('/api/companies/discover', methods=['GET'])
@token_required
def discover_companies(current_user, role, user_id):
    if role != 'investor':
        return jsonify({"message": "Access restricted"}), 403

    try:
        conn = sqlite3.connect('assets/users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, com_category FROM users WHERE role != 'investor' AND role != 'admin'")
        companies = cursor.fetchall()
        conn.close()

        company_data = []
        pred_conn = sqlite3.connect('assets/database.db')
        pred_cursor = pred_conn.cursor()

        for comp in companies:
            c_id, c_username, c_category = comp
            pred_cursor.execute("""
                SELECT int_rate, default_rate, sus_score 
                FROM predictions 
                WHERE user_id = ? 
                ORDER BY created_at DESC LIMIT 1
            """, (c_id,))
            
            latest = pred_cursor.fetchone()
            score = 0
            defs = 0.0
            ints = 0.0
            
            if latest:
                i_rate, d_rate, s_score = latest
                i_rate = i_rate if i_rate is not None else 0
                d_rate = d_rate if d_rate is not None else 0
                s_score = s_score if s_score is not None else 0
                
                score = esgatescoref(i_rate, d_rate, s_score)
                defs = d_rate
                ints = i_rate

            company_data.append({
                "id": c_id,
                "name": c_username.capitalize(),
                "category": c_category if c_category else "Unspecified",
                "esg_score": score,
                "default_rate": f"{defs*100:.1f}%",
                "int_rate": f"{ints*100:.2f}%",
                "compliance": "CSRD Compliant" if score > 75 else "In Progress"
            })

        pred_conn.close()
        return jsonify({"companies": company_data}), 200

    except Exception as e:
        print(f"Error in discover_companies: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route('/predict_default', methods=['POST'])
@token_required
def predict_default(current_user, role,user_id):
    data_def = request.get_json()
    if not data_def:
        return jsonify({"error": "No JSON data received"}), 400


    # user_conn = sqlite3.connect('assets/users.db')
    # user_cursor = user_conn.cursor()
    # user_cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    # user_row = user_cursor.fetchone()

    # if user_row is None:
    #     user_conn.close()
    #     return jsonify({"error": "User not found"}), 400 
    # user_id = user_row[0]
    # user_conn.close() 


    try:
        default_rate = predict_default_sme(data_def)
        if default_rate>=0.42:
            default_rate = default_rate-0.3

        else:
            default_rate = default_rate -0.17

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    ok, reason = validate_prediction_value(default_rate, "default_rate", 0.0, 1.0)
    if not ok:
        return jsonify({"error": f"Invalid model output: {reason}"}), 500

    try:
        u_conn = sqlite3.connect('assets/users.db')
        u_cursor = u_conn.cursor()
        u_cursor.execute("SELECT com_category FROM users WHERE id=?", (user_id,))
        row = u_cursor.fetchone()
        user_category = row[0] if row else "Unspecified"
        u_conn.close()
    except:
        user_category = "Unspecified"


    #New UPSERT
    try:
        save_prediction_metric(
            user_id,
            current_user,
            user_category,  # this was the issue the whole time
            'default_rate',
            float(default_rate)
        )
    except Exception as e:
        return jsonify({"error": f"Failed to save prediction: {str(e)}"}), 500

    gc.collect()
    torch.cuda.empty_cache()
    return jsonify({'default_rate': float(default_rate)}), 200


@app.route('/predict_int_rate', methods=['POST'])
@token_required
def predict_int_rate(current_user, role, user_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    required = ['operation_years','revenue','office_own','team_exp','loan_amt','default_hist','cred_hist_len','repayment_status']
    for k in required:
        if k not in data:
            return jsonify({"error": f"Missing required field: {k}"}), 400

    try:
        int_rate = predict_company_metrics(
            model_reg_loaded, scaler, derived_feature_info, feature_names_order_reg,
            numerical_cols_to_scale_fit, device, 'rate',
            data['operation_years'], data['revenue'], data['office_own'], data['team_exp'],
            data['loan_amt'], data['default_hist'], data['cred_hist_len'], data.get('repayment_status', 0)
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    ok, reason = validate_prediction_value(int_rate, "int_rate", -0.5, 5.0)
    if not ok:
        return jsonify({"error": f"Invalid model output: {reason}"}), 500


    try:
        u_conn = sqlite3.connect('assets/users.db')
        u_cursor = u_conn.cursor()
        u_cursor.execute("SELECT com_category FROM users WHERE id=?", (user_id,))
        row = u_cursor.fetchone()
        user_category = row[0] if row else "Unspecified"
        u_conn.close()
    except:
        user_category = "Unspecified"

    try:
        save_prediction_metric(
            user_id,
            current_user,
            user_category,   
            'int_rate',
            float(int_rate)
        )
    except Exception as e:
        # Logs the error but doesn't crash the whole request if save fails
        print(f"Error saving prediction: {e}")

    
    int_rate += 0.05

    gc.collect()
    torch.cuda.empty_cache()
    return jsonify({'int_rate': float(int_rate)}), 200

def esgatescoref(int_rate, default_rate, sus_score):
    sustainability_component = (sus_score+0.1) * 10    
    default_risk_component = (1 - default_rate) * 100
    MAX_ACCEPTABLE_RATE = 0.7
    normalized_rate = min(max(int_rate, 0.0), MAX_ACCEPTABLE_RATE) 
    interest_rate_component = (1 - (normalized_rate / MAX_ACCEPTABLE_RATE)) * 100
    WEIGHT_SUS = 0.31
    WEIGHT_DEF = 0.41
    WEIGHT_INT = 0.31
    
    final_score = (sustainability_component * WEIGHT_SUS) + \
                  (default_risk_component * WEIGHT_DEF) + \
                  (interest_rate_component * WEIGHT_INT)
                  
    return min(int(final_score),100)

### =======================================================================

# I spent a significant portion of my time on this
def sus_prompt(metrics, averages):
    """Builds a detailed, structured prompt for the Mistral API."""
    prompt = f"""
    You are an expert ESG (Environmental, Social, and Governance) analyst. Your task is to evaluate a company's sustainability performance based on provided data and sector averages.

    **Company's Performance Data:**
    - Energy Efficiency: {metrics.get('energy_efficiency')} MWh/unit
    - Carbon Intensity: {metrics.get('carbon_intensity')} tCO2e/unit
    - Water Usage: {metrics.get('water_usage')} Liters/unit

    **Sector Averages (Benchmark):**
    - Energy Efficiency: {averages.get('energy_efficiency')} MWh/unit
    - Carbon Intensity: {averages.get('carbon_intensity')} tCO2e/unit
    - Water Usage: {averages.get('water_usage')} Liters/unit

    **Your Task:**
    1.  **Calculate a Sustainability Score:** Provide a single score from 0 (terrible) to 10 (excellent).
        - For all metrics, a **lower** value is better.
        - A company performing close to the sector average should score around 5.
        - A company performing significantly better (e.g., half the average usage) should score closer to 10.
        - A company performing significantly worse (e.g., double the average usage) should score closer to 0.
        - **Critically, do not give a perfect 10 or a 0 unless the data is truly exceptional or abysmal.** Penalize any single metric that is extremely poor, as it indicates a significant risk area.

    2.  **Provide Qualitative Analysis:** Based on the comparison, write a brief analysis.
        - **Summary:** A one-sentence overview of the company's performance.
        - **Strengths:** A bulleted list of 1-2 areas where the company excels.
        - **Weaknesses:** A bulleted list of 1-2 areas where the company is underperforming.
        - **Recommendations:** A bulleted list of 1-2 actionable recommendations for improvement.

    **Output Format:**
    You MUST return your response as a single, valid JSON object. Do not include any text, notes, or explanations outside of the JSON structure.

    {{
     "sus_score": <float>,
     "summary": "<string>",
      "strengths": ["<string>"],
      "weaknesses": ["<string>"],
      "recommendations": ["<string>"]
      }}
    """
    return prompt
def parse_mistral_api_output(result):
    try:
        content_string = result.get("choices", [{}])[0].get("message", {}).get("content")
        
        if content_string is None:
            print('Error parsing Mistral output: Content key was missing or None')
            content_string = result.get("choices", [{}])[0].get("text")
            # return {"error": "AI response was missing 'content' key"}
        if content_string is None:
            print(f'Error parsing Mistral output: Context variable was missing or None')
            print(f'Full result structure: {json.dumps(result,indent=2)}')
            return {'error': 'AI response was missing "context" key'}

        content_strip = content_string.strip()
        
        match = re.search(r"```json\s*(\{.*\})\s*```", content_strip, re.DOTALL)
        if match:
            json_text = match.group(1)
        else:
            json_text = content_strip 

        if not json_text:
            print('Error parsing Mistral output: Received an EMPTY string')
            return {'error': 'AI returned an empty response'}

        return json.loads(json_text)
    
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Error parsing Mistral output: {e}")
        print(f"Raw content received: {content_string}") 
        return {"error": "Failed to parse AI response. The response might be malformed."}
    

# ADDED NEW BLACK MAGIC DEBUDDING STUFF TO FIX THE ISSUE OF SUS_SCORE NOT APPEARING

@app.route('/sustainability_prediction', methods=['POST'])
@token_required
def sustainability_prediction_endpoint(current_user, role,user_id):
    print("\n--- ENTERING /sustainability_prediction ---")
    metrics_dict = request.get_json()
    if not metrics_dict:
        print("ERROR: No JSON body received.")
        return jsonify({"error": "Missing or invalid JSON body"}), 400
    print(f"Received metrics: {metrics_dict}")

    sector_averages = {
        'energy_efficiency': 4.5,
        'carbon_intensity': 1.0,
        'water_usage': 3500
    }

    HEADERS = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-small-3.2-24b-instruct:free",
        "messages": [
            {"role": "user", "content": sus_prompt(metrics_dict, sector_averages)}
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"},
        "max_tokens": 500
    }

    # --- DUMMY RESPONSE DEFINITION ---

    DUMMY_RESULT = {
        "sus_score": 3.0, # A neutral, placeholder score
        "summary": "AI Analysis is currently unavailable. The score below is a neutral placeholder. You may try again shortly.",
        "strengths": [
            "Resilience: Automatic fallback to placeholder analysis.",
            "Core operational metrics were successfully processed."
        ],
        "weaknesses": [
            "Tailored, data-driven analysis from the AI model could not be generated.",
            "Benchmarking against sector averages is currently unavailable."
        ],
        "recommendations": [
            "Review your submission metrics manually.",
            "Please try the prediction again in a few minutes, as external services may be temporarily overloaded."
        ]
    }
    # --- END DUMMY RESPONSE DEFINITION ---

    parsed_result = None

    try:
        print("Calling Mistral  for sustainability...")
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=HEADERS, json=payload)
        response.raise_for_status() # Raises an error
        print("Mistral API call successful.")

        result = response.json()
        parsed_result = parse_mistral_api_output(result)

        # Check for error from AI response
        if "error" in parsed_result or "sus_score" not in parsed_result:
            print(f"WARNING: API call succeeded but parsing failed or AI returned an error. Using DUMMY data.")
            print(f"Parsing/AI Error Details: {parsed_result.get('error', 'sus_score missing')}")
            # This is going to save our demo
            parsed_result = DUMMY_RESULT
            print(f"WARNING: Using DUMMY summary: {DUMMY_RESULT['summary']}")

    except (requests.exceptions.RequestException, Exception) as e:
        print(f"ERROR: Mistral API request failed or internal processing error: {e}")
        # Plug in the dummy result
        parsed_result = DUMMY_RESULT
        print("DUMMY values plugged in due to API failure.")

    # This block executes whether the result is real or dummy.
    score_to_save = parsed_result.get('sus_score')
    
    # Ensure score_to_save is convertible to float
    try:
        score_to_save_float = float(score_to_save)
    except (ValueError, TypeError):
        print(f"WARNING: Score '{score_to_save}' is not a valid number. Skipping save.")
        score_to_save_float = None

    if score_to_save_float is not None:
        try:
            if user_id:
                # Get the category for this user
                user_category = "Unspecified"
                try:
                    u_conn = sqlite3.connect('assets/users.db')
                    u_cursor = u_conn.cursor()
                    u_cursor.execute("SELECT com_category FROM users WHERE id=?", (user_id,))
                    row = u_cursor.fetchone()
                    user_category = row[0] if row else "Unspecified"
                    u_conn.close()
                except Exception as cat_e:
                    print(f"Warning: Could not fetch user category for saving: {cat_e}")

                print(f"Found user_id: {user_id}. Calling save_prediction_metric with score: {score_to_save_float}...")
                save_prediction_metric(
                    user_id,
                    current_user,
                    user_category,
                    'sus_score',
                    score_to_save_float
                )
            else:
                print("ERROR: User ID is missing, cannot save sustainability score.")

        except Exception as e:
            print(f"ERROR during save process: Failed to save sus_score to database: {e}")
            traceback.print_exc()
    else:
        print("WARNING: Cannot save score because 'sus_score' was invalid or missing after processing.")


    print("--- EXITING /sustainability_prediction ---")
    return jsonify(parsed_result)








@app.route('/user_predictions/<company>', methods=['GET'])
@token_required
def user_predictions(current_user, role,user_id, company):
    try:
        conn = sqlite3.connect('assets/database.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT int_rate, default_rate, sus_score, notes, created_at
            FROM predictions
            WHERE user_id = ? AND lower(company_name) = ?
            ORDER BY created_at ASC
        ''', (user_id, company.lower())) 
        
        rows = cursor.fetchall()
        conn.close()

        formatted = []
        for r in rows:
            formatted.append({
                "int_rate": None if r[0] is None else float(r[0]),
                "default_rate": None if r[1] is None else float(r[1]),
                "sus_score": None if r[2] is None else float(r[2]),
                "notes": r[3],
                "created_at": r[4]
            })
        return jsonify({"predictions": formatted}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -------------------------- AI Overview --------------------------
@app.route('/company_summary', methods=['POST'])
@token_required
def company_summary(current_user, role,user_id): # CURRENT_USER AND ROLE SHALL BE IMPLEMENTED IN THE FUTURE
    data = request.get_json()
    int_rate = data.get('int_rate')
    def_rate = data.get("default_rate")
    sus_score = data.get("sus_score")
    
    try:
        esgatescore = esgatescoref(int_rate,def_rate,sus_score)
    except Exception as e:
        print(f'Error calculating ESGate Score: ', {e})
        esgatescore = None

    metrics_dict = {
        "int_rate": int_rate,
        "default_rate": def_rate,
        "sus_score": sus_score,
        'esgatescore' :esgatescore,
        "notes": data.get("notes", "")
    }

    summary_result = get_mistral_summary(metrics_dict)

    return jsonify({"mistral_summary": summary_result,"esgatescore" :esgatescore}), 200


@app.route('/predict_all_and_save', methods=['POST'])
@token_required
def predict_all_and_save(current_user, role,user_id):
    payload = request.get_json()
    int_rate = payload.get('int_rate')
    default_rate = payload.get('default_rate')
    sus_score = payload.get('sus_score')
    company_name = str(payload.get('company_name', '')).lower()

    # Basic validation
    if int_rate is not None:
        ok, reason = validate_prediction_value(int_rate, "int_rate", -0.5, 5.0)
        if not ok:
            return jsonify({"error": f"Invalid int_rate: {reason}"}), 400
    if default_rate is not None:
        ok, reason = validate_prediction_value(default_rate, "default_rate", 0.0, 1.0)
        if not ok:
            return jsonify({"error": f"Invalid default_rate: {reason}"}), 400
    if sus_score is not None:
        ok, reason = validate_prediction_value(sus_score, "sus_score", 0.0, 100.0)
        if not ok:
            return jsonify({"error": f"Invalid sus_score: {reason}"}), 400

    conn = sqlite3.connect('assets/users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    if user_row is None:
        conn.close()
        return jsonify({"error": "User not found"}), 400
    user_id = user_row[0]
 
# --- MISTRAL API CONFIGURATION ---
# Optional enviroment variables
# api_key = os.environ.get("MISTRAL_API_KEY")
# model = "mistral-large-latest"
# client = MistralClient(api_key=api_key)



def get_mistral_tip():
    payload = {
            "model": "mistralai/mistral-small-3.2-24b-instruct:free",
            "messages": [
                {"role": "user", "content": "Generate a short, actionable, innovative and inspiring ESG tip for a company dashboard. The tip should be a single sentence."}
            ],
            "temperature": 0.7,
            "max_tokens": 550 
        }
        
    response = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload)
    
    if response.status_code != 200:
        return {"error": f"Mistral API error: {response.text}"}
    
    result = response.json()
    
    try:
        return result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    except Exception as e:
        return {"error": f"Failed to parse tip: {str(e)}"}

@app.route('/get-esg-tip', methods=['POST'])
def get_esg_tip():
    """
    API endpoint that the frontend will call to get a new ESG tip.
    """
    try:
        # pass parameters from the frontend if needed
        # data = request.get_json() 
        # user_profile = data.get('profile') 

        ai_tip = get_mistral_tip()
        
        return jsonify({'tip': ai_tip})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Failed to generate tip'}), 500
# ------------------------ Last additions ------------------------
@app.route('/get_gemini_news')
def get_gemini_news():
    sys_prompt = '''
    You are a **GROUNDED NEWS SUMMARIZER AND OUTPUT FORMATTER**. 
    Your primary task is to find a news article and format its details.
    
    1. **Search Focus**: You MUST use your search tool to find one recent, highly relevant news article ONLY about **EU ESG/CSRD rules affecting SMEs**.
    2. **Grounding**: Once you have a source, you MUST generate the headline and summary **BASED SOLELY ON THE CONTENT OF THAT FOUND ARTICLE**.
    3. **URL Priority**: You MUST copy the URL **DIRECTLY** from the search result. DO NOT modify or make up the URL.
    4. **Output Format**: The output MUST adhere to the following strict, asterisk-separated format:
       [3-5 word news headline]*[single, short summary (under 25 words)]*[REAL, full URL from the source]

    CRITICAL RULES:
    - The link MUST be a complete URL starting with 'https://' or 'http://'.
    - Do not use any asterisks (*) in the headline or summary.
    - Do not include any introductory text, markdown, or concluding remarks. The response must be ONLY the formatted string.
    '''
    # --------------------------------------------------------------------------

    prompt = news_prompts[random.randrange(0,29)]
    gen_config = types.GenerateContentConfig(system_instruction=sys_prompt)
    response = client.models.generate_content(model=model_n,contents=[prompt],config=gen_config)
    output = response.text

    if output:
        parts = output.split('*')
        print('RAW OUTPUT', parts)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= 3:
            headline = parts[0]
            summary = parts[1]
            link = parts[2]
            
            # Defensive URL correction:
            if not link.startswith(('http://', 'https://')):
                link = 'https://' + link.lstrip('/')
                print(f'WARNING: Prepending "https://" to partial URL: {link}')
            
            # Final domain validation check: (Using the imported 're')
            if re.search(r'\.(com|org|net|co|gov|eu|int)/', link):
                print('SUCCESSFULLY parsed the output')
                return jsonify({'title':headline,'summary':summary.strip(),'link':link.strip()}), 200
            else:
                print(f'ERROR: URL failed final domain validation: {link}')
                # Return an error for an invalid-looking URL
                return jsonify({'error':'GEMINI NEWS ERROR: Invalid URL generated by model.'}), 500

        else:
            print(f'ERROR WHILE PARSING FROM GEMINI. Expected 3 parts, got {len(parts)}.\nRaw Parts: {parts}')
            return jsonify({'error':'GEMINI NEWS ERROR: Output did not contain expected delimiters.'}), 500
    
    return jsonify({'error':'GEMINI NEWS ERROR: No output received.'}), 500



# -------------------------- Run Server --------------------------
if __name__ == '__main__':
    app.run(debug=True)
