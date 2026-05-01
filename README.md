
# LegalAI - Differentiated Case Flow Management System

A comprehensive full-stack application for intelligent case flow management with AI-powered assistance, advanced scheduling, and analytics capabilities. Built with FastAPI (backend), MongoDB, and vanilla HTML/CSS/JavaScript (frontend).

## 🎯 Overview

LegalAI is an intelligent case flow management system designed to streamline case handling with:
- **AI-Powered Assistance**: BNS (Bayesian Network System) models for intelligent case classification and suggestions
- **Smart Scheduling**: Advanced scheduling algorithms for optimal case prioritization and hearing management
- **Analytics & Reporting**: Comprehensive dashboards and reports for case metrics and system health
- **Document Management**: Secure handling and processing of case documents
- **Multi-Role Access Control**: Support for different user roles (Admin, Judge, Clerk) with role-based permissions
- **Audit Trail**: Complete audit logging for compliance and accountability

## ✨ Features

### Case Management
- Create, read, update, and delete cases
- Case categorization and priority assignment
- Document upload and management
- Case history and audit tracking

### AI & NLP Services
- BNS Ensemble Model for intelligent case classification
- Natural Language Processing for document analysis
- Smart suggestions based on case content
- Automated case routing recommendations

### Scheduling
- Intelligent hearing scheduling
- Priority-based case allocation
- Conflict resolution
- Calendar integration

### Analytics & Dashboards
- Real-time case statistics
- Performance metrics
- System health monitoring
- Advanced reporting capabilities

### User Management
- Role-based access control (RBAC)
- Multi-factor authentication support
- User audit trails
- Department/organization management

### Notification System
- Email notifications
- SMS alerts
- In-app notifications
- Customizable notification rules

## 🛠 Tech Stack

### Backend
- **Framework**: FastAPI 0.104.1
- **Database**: MongoDB with Motor (async driver) and Beanie ORM
- **Authentication**: JWT with python-jose
- **NLP**: NLTK, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Report Generation**: ReportLab
- **Testing**: Pytest, pytest-asyncio
- **Server**: Uvicorn
- **API Gateway**: FastAPI CORS middleware
- **Logging**: Python logging

### Frontend
- **Language**: HTML5, CSS3, Vanilla JavaScript
- **Architecture**: Client-side MVC pattern
- **HTTP Client**: Fetch API / Axios
- **Styling**: Pure CSS with responsive design
- **DOM Manipulation**: Vanilla JavaScript

## 📁 Project Structure

```
JustiFlow-Differentiated_caseflow_management--main/
├── backend/                                  # Python backend
│   ├── app/
│   │   ├── core/                           # Core configuration
│   │   │   ├── config.py                   # Settings and environment variables
│   │   │   ├── database.py                 # Database connections
│   │   │   ├── security.py                 # Authentication and authorization
│   │   │   └── exceptions.py               # Custom exceptions
│   │   ├── routers/                        # API endpoints
│   │   │   ├── auth.py                     # Authentication endpoints
│   │   │   ├── cases.py                    # Case management endpoints
│   │   │   ├── users.py                    # User management endpoints
│   │   │   ├── analytics.py                # Analytics endpoints
│   │   │   ├── reports.py                  # Reporting endpoints
│   │   │   ├── schedule.py                 # Scheduling endpoints
│   │   │   └── nlp.py                      # NLP service endpoints
│   │   ├── services/                       # Business logic
│   │   │   ├── ai_service.py               # AI service orchestration
│   │   │   ├── dcm_service.py              # Case flow management service
│   │   │   ├── case_ingestion_service.py   # Case data ingestion
│   │   │   ├── smart_scheduling_service.py # Scheduling logic
│   │   │   ├── analytics_service.py        # Analytics calculations
│   │   │   ├── audit_service.py            # Audit logging
│   │   │   ├── email_service.py            # Email notifications
│   │   │   ├── sms_service.py              # SMS notifications
│   │   │   └── nlp/                        # NLP models
│   │   │       ├── bns_ensemble_model.py   # Bayesian Network ensemble
│   │   │       └── enhanced_bns_service.py # Enhanced BNS service
│   │   └── templates/                      # Email templates
│   ├── main_mongodb.py                     # Application entry point
│   ├── config.py                           # Configuration loader
│   ├── requirements.txt                    # Python dependencies
│   └── requirements-production.txt         # Production dependencies
├── frontend/                                # Vanilla HTML/CSS/JS frontend
│   ├── index.html                          # Main application page
│   ├── css/                                # Stylesheets
│   │   ├── styles.css                      # Main stylesheet
│   │   ├── responsive.css                  # Responsive design styles
│   │   └── components.css                  # Component styles
│   ├── js/                                 # JavaScript files
│   │   ├── main.js                         # Application entry point
│   │   ├── api.js                          # API integration
│   │   ├── auth.js                         # Authentication logic
│   │   ├── ui.js                           # UI components and handlers
│   │   └── utils.js                        # Utility functions
│   └── pages/                              # HTML pages
│       ├── login.html                      # Login page
│       ├── dashboard.html                  # Main dashboard
│       ├── cases.html                      # Case management
│       ├── schedule.html                   # Scheduling page
│       ├── reports.html                    # Reports page
│       └── users.html                      # User management
├── models/                                  # ML models
│   └── enhanced_model_info.json            # Model metadata
└── requirements.txt                        # Root dependencies
```

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **MongoDB**: 4.4 or higher
- **pip**: Python package manager (comes with Python)
- **Web Browser**: Modern browser with ES6 support (Chrome, Firefox, Safari, Edge)

## 🚀 Installation

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup


1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Open in browser** or **serve with a local HTTP server**:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Or use any local server (node-http-server, PHP, etc.)
   ```
   Then open `http://localhost:8000` in your browser.

## ⚙️ Configuration

### Backend Configuration

Create a `.env` file in the `backend/` directory:

```env
# Application
APP_NAME=DCM System with BNS Assist
APP_VERSION=1.0.0
DEBUG=True
ENVIRONMENT=development
PORT=8001

# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=dcm_system
MONGODB_USERNAME=your_username
MONGODB_PASSWORD=your_password
MONGODB_CLUSTER=your_cluster

# JWT
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Email Service
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# SMS Service
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1234567890

# Logging
LOG_LEVEL=INFO
```

## 🏃 Running the Application

### Start MongoDB

```bash
# Using Docker (recommended)
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or start MongoDB service if installed locally
mongod
```

### Start Backend

```bash
cd backend
source venv/bin/activate  # Windows: venv\Scripts\activate
uvicorn main_mongodb:app --reload --host 0.0.0.0 --port 8001
```

Backend will be available at: `http://localhost:8001`

### Start Frontend

```bash
cd frontend
python -m http.server 8000
```

Frontend will be available at: `http://localhost:8000`

### API Documentation

Once the backend is running, access the API documentation at:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`



### Training Models

To train the BNS ensemble model:

```bash
cd backend
python train_enhanced_bns_model.py
```

To train with database cases:

```bash
python train_with_database_cases.py
```

## 📊 Database Schema

### Key Collections (MongoDB)

- **users**: User accounts and profiles
- **cases**: Case records
- **documents**: Uploaded documents
- **schedules**: Hearing schedules
- **audit_logs**: System audit trails
- **analytics_snapshots**: Analytics data points
- **notifications**: User notifications

## 🔐 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Role-based access control (RBAC)
- Audit logging for all operations
- CORS protection
- SQL/NoSQL injection prevention
- Secure file upload validation

## 📝 API Endpoints Summary

### Authentication
- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `POST /auth/refresh` - Refresh token

### Cases
- `GET /cases` - List all cases
- `POST /cases` - Create new case
- `GET /cases/{id}` - Get case details
- `PUT /cases/{id}` - Update case
- `DELETE /cases/{id}` - Delete case

### AI & NLP
- `POST /nlp/classify` - Classify case documents
- `POST /nlp/suggestions` - Get AI suggestions
- `POST /nlp/analyze` - Analyze document content

### Scheduling
- `GET /schedule/hearings` - List scheduled hearings
- `POST /schedule/auto-schedule` - Generate optimal schedule
- `PUT /schedule/hearings/{id}` - Update hearing

### Analytics
- `GET /analytics/dashboard` - Get dashboard metrics
- `GET /analytics/reports` - Generate reports
- `GET /analytics/health` - System health status

### Users
- `GET /users` - List users
- `POST /users` - Create user
- `PUT /users/{id}` - Update user
- `DELETE /users/{id}` - Delete user

## 📄 License

This project is proprietary and confidential.

## 📞 Support

For support and questions, please contact.


---


