/**
 * JustiFlow - DCM System JavaScript
 * Vanilla JavaScript implementation
 */

// ========================================
// API Configuration
// ========================================
const API_BASE_URL = 'http://localhost:8000/api';

// Simple API service
const api = {
    async request(endpoint, options = {}) {
        const url = `${API_BASE_URL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add auth token if available
        const token = localStorage.getItem('token');
        if (token) {
            config.headers['Authorization'] = `Bearer ${token}`;
        }

        try {
            const response = await fetch(url, config);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    },

    post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
};

// ========================================
// Auth Service
// ========================================
const auth = {
    user: null,
    token: null,

    async login(email, password) {
        try {
            // For demo, simulate login - in production, call API
            // const data = await api.post('/auth/login', { email, password });
            
            // Simulated response for demo
            if (email && password) {
                this.token = 'demo-token-' + Date.now();
                this.user = {
                    id: 1,
                    email: email,
                    full_name: 'Admin User',
                    role: 'ADMIN'
                };
                
                localStorage.setItem('token', this.token);
                localStorage.setItem('user', JSON.stringify(this.user));
                
                return { success: true, user: this.user };
            }
            throw new Error('Invalid credentials');
        } catch (error) {
            console.error('Login error:', error);
            throw error;
        }
    },

    logout() {
        this.token = null;
        this.user = null;
        localStorage.removeItem('token');
        localStorage.removeItem('user');
    },

    checkAuth() {
        const token = localStorage.getItem('token');
        const userStr = localStorage.getItem('user');
        
        if (token && userStr) {
            this.token = token;
            this.user = JSON.parse(userStr);
            return true;
        }
        return false;
    },

    getUser() {
        return this.user;
    },

    isAdmin() {
        return this.user && this.user.role === 'ADMIN';
    },

    isJudge() {
        return this.user && this.user.role === 'JUDGE';
    },

    isClerk() {
        return this.user && this.user.role === 'CLERK';
    }
};

// ========================================
// Toast Notifications
// ========================================
const toast = {
    container: null,

    init() {
        this.container = document.getElementById('toast-container');
    },

    show(message, type = 'info', duration = 4000) {
        const toastEl = document.createElement('div');
        toastEl.className = `toast ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-times-circle',
            warning: 'fa-exclamation-circle',
            info: 'fa-info-circle'
        };

        toastEl.innerHTML = `
            <i class="fas ${icons[type]}"></i>
            <span class="toast-message">${message}</span>
            <button class="toast-close"><i class="fas fa-times"></i></button>
        `;

        this.container.appendChild(toastEl);

        // Close button
        toastEl.querySelector('.toast-close').addEventListener('click', () => {
            this.remove(toastEl);
        });

        // Auto remove
        setTimeout(() => {
            this.remove(toastEl);
        }, duration);
    },

    remove(toastEl) {
        toastEl.style.animation = 'toastSlide 0.3s ease-out reverse';
        setTimeout(() => {
            if (toastEl.parentNode) {
                toastEl.parentNode.removeChild(toastEl);
            }
        }, 300);
    },

    success(message) {
        this.show(message, 'success');
    },

    error(message) {
        this.show(message, 'error');
    },

    warning(message) {
        this.show(message, 'warning');
    },

    info(message) {
        this.show(message, 'info');
    }
};

// ========================================
// Modal Manager
// ========================================
const modal = {
    container: null,
    title: null,
    body: null,
    footer: null,
    closeBtn: null,
    cancelBtn: null,
    confirmBtn: null,
    onConfirm: null,

    init() {
        this.container = document.getElementById('modal-container');
        this.title = document.getElementById('modal-title');
        this.body = document.getElementById('modal-body');
        this.footer = document.getElementById('modal-footer');
        this.closeBtn = document.getElementById('modal-close');
        this.cancelBtn = document.getElementById('modal-cancel');
        this.confirmBtn = document.getElementById('modal-confirm');

        // Event listeners
        this.closeBtn.addEventListener('click', () => this.hide());
        this.cancelBtn.addEventListener('click', () => this.hide());
        this.container.querySelector('.modal-backdrop').addEventListener('click', () => this.hide());
        
        this.confirmBtn.addEventListener('click', () => {
            if (this.onConfirm) {
                this.onConfirm();
            }
            this.hide();
        });
    },

    show(options) {
        this.title.textContent = options.title || 'Modal';
        this.body.innerHTML = options.body || '';
        
        if (options.showFooter === false) {
            this.footer.classList.add('hidden');
        } else {
            this.footer.classList.remove('hidden');
            this.confirmBtn.textContent = options.confirmText || 'Confirm';
            this.cancelBtn.textContent = options.cancelText || 'Cancel';
        }

        this.onConfirm = options.onConfirm;
        this.container.classList.remove('hidden');
    },

    hide() {
        this.container.classList.add('hidden');
        this.body.innerHTML = '';
        this.onConfirm = null;
    },

    confirm(options) {
        this.show({
            title: options.title,
            body: options.message,
            confirmText: options.confirmText || 'Confirm',
            cancelText: options.cancelText || 'Cancel',
            onConfirm: options.onConfirm
        });
    },

    alert(options) {
        this.show({
            title: options.title,
            body: options.message,
            showFooter: false
        });
        
        // Auto close after delay
        setTimeout(() => this.hide(), options.duration || 2000);
    }
};

// ========================================
// Page Router
// ========================================
const router = {
    currentPage: 'dashboard',
    pages: [
        'dashboard',
        'ai-dashboard',
        'cases',
        'documents',
        'hearings',
        'scheduling',
        'priority-scheduling',
        'reports',
        'users',
        'settings'
    ],

    init() {
        this.bindNavigation();
        this.bindQuickActions();
    },

    navigate(page) {
        if (!this.pages.includes(page)) {
            console.error('Page not found:', page);
            return;
        }

        // Hide all pages
        document.querySelectorAll('.content-page').forEach(p => {
            p.classList.remove('active');
        });

        // Show target page
        const targetPage = document.getElementById(`page-${page}`);
        if (targetPage) {
            targetPage.classList.add('active');
            this.currentPage = page;
            this.updateNavState(page);
            this.updatePagePermissions(page);
        }
    },

    updateNavState(page) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.page === page) {
                item.classList.add('active');
            }
        });
    },

    updatePagePermissions(page) {
        // Show/hide admin-only pages based on user role
        const adminPages = document.querySelectorAll('.admin-only-page');
        
        if (auth.isAdmin()) {
            adminPages.forEach(p => p.classList.remove('hidden'));
        } else {
            // Hide admin pages for non-admin users
            if (['priority-scheduling', 'reports', 'users'].includes(page)) {
                this.navigate('dashboard');
            }
        }
    },

    bindNavigation() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const page = item.dataset.page;
                
                // Check permissions
                if (item.classList.contains('admin-only') && !auth.isAdmin()) {
                    toast.warning('You do not have permission to access this page');
                    return;
                }

                this.navigate(page);
                this.closeMobileSidebar();
            });
        });
    },

    bindQuickActions() {
        document.querySelectorAll('.action-card').forEach(btn => {
            btn.addEventListener('click', () => {
                const page = btn.dataset.page;
                const action = btn.dataset.action;
                
                this.navigate(page);
                
                if (action === 'new') {
                    // Trigger new case modal
                    this.showNewCaseModal();
                }
            });
        });
    },

    closeMobileSidebar() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');
        
        if (window.innerWidth < 1024) {
            sidebar.classList.remove('open');
            overlay.classList.add('hidden');
        }
    },

    showNewCaseModal() {
        modal.show({
            title: 'Create New Case',
            body: `
                <form id="new-case-form">
                    <div class="form-group" style="margin-bottom: 1rem;">
                        <label class="form-label">Case Title</label>
                        <input type="text" class="form-input" id="case-title" placeholder="Enter case title" required>
                    </div>
                    <div class="form-group" style="margin-bottom: 1rem;">
                        <label class="form-label">Case Type</label>
                        <select class="form-select" id="case-type" style="width: 100%;" required>
                            <option value="">Select type</option>
                            <option value="CIVIL">Civil</option>
                            <option value="CRIMINAL">Criminal</option>
                            <option value="FAMILY">Family</option>
                            <option value="CORPORATE">Corporate</option>
                        </select>
                    </div>
                    <div class="form-group" style="margin-bottom: 1rem;">
                        <label class="form-label">Priority</label>
                        <select class="form-select" id="case-priority" style="width: 100%;" required>
                            <option value="MEDIUM">Medium</option>
                            <option value="HIGH">High</option>
                            <option value="CRITICAL">Critical</option>
                            <option value="LOW">Low</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Description</label>
                        <textarea class="form-input" id="case-description" rows="4" placeholder="Enter case description"></textarea>
                    </div>
                </form>
            `,
            confirmText: 'Create Case',
            onConfirm: () => {
                const title = document.getElementById('case-title').value;
                const type = document.getElementById('case-type').value;
                const priority = document.getElementById('case-priority').value;
                const description = document.getElementById('case-description').value;

                if (!title || !type) {
                    toast.error('Please fill in all required fields');
                    return;
                }

                // Simulate case creation
                toast.success('Case created successfully!');
                // In production, call API to create case
            }
        });
    }
};

// ========================================
// Sidebar Manager
// ========================================
const sidebar = {
    init() {
        this.bindEvents();
    },

    bindEvents() {
        const menuToggle = document.getElementById('menu-toggle');
        const sidebarClose = document.getElementById('sidebar-close');
        const overlay = document.getElementById('sidebar-overlay');
        const sidebarEl = document.getElementById('sidebar');

        menuToggle.addEventListener('click', () => {
            sidebarEl.classList.add('open');
            overlay.classList.remove('hidden');
        });

        sidebarClose.addEventListener('click', () => {
            this.close();
        });

        overlay.addEventListener('click', () => {
            this.close();
        });
    },

    close() {
        const sidebarEl = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');
        
        sidebarEl.classList.remove('open');
        overlay.classList.add('hidden');
    }
};

// ========================================
// Dropdown Manager
// ========================================
const dropdowns = {
    init() {
        this.bindEvents();
    },

    bindEvents() {
        // Notification dropdown
        const notifBtn = document.getElementById('notification-btn');
        const notifMenu = document.getElementById('notification-menu');
        
        notifBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            notifMenu.classList.toggle('hidden');
            document.getElementById('user-menu').classList.add('hidden');
        });

        // User dropdown
        const userBtn = document.getElementById('user-btn');
        const userMenu = document.getElementById('user-menu');
        
        userBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            userMenu.classList.toggle('hidden');
            notifMenu.classList.add('hidden');
        });

        // Close dropdowns when clicking outside
        document.addEventListener('click', () => {
            notifMenu.classList.add('hidden');
            userMenu.classList.add('hidden');
        });

        // Prevent dropdown close when clicking inside
        notifMenu.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        userMenu.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }
};

// ========================================
// Login Handler
// ========================================
const loginHandler = {
    form: null,

    init() {
        this.form = document.getElementById('login-form');
        this.bindEvents();
    },

    bindEvents() {
        this.form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = document.getElementById('login-email').value;
            const password = document.getElementById('login-password').value;

            if (!email || !password) {
                toast.error('Please enter email and password');
                return;
            }

            try {
                const result = await auth.login(email, password);
                
                if (result.success) {
                    toast.success('Login successful!');
                    this.showMainApp(result.user);
                }
            } catch (error) {
                toast.error('Login failed. Please check your credentials.');
            }
        });
    },

    showMainApp(user) {
        const loginPage = document.getElementById('login-page');
        const mainApp = document.getElementById('main-app');
        
        loginPage.classList.add('hidden');
        mainApp.classList.remove('hidden');

        // Update user info
        document.querySelector('.user-name').textContent = user.full_name;
        document.querySelector('.user-role').textContent = user.role;

        // Update sidebar nav permissions
        this.updateNavPermissions(user.role);
    },

    updateNavPermissions(role) {
        const adminNavItems = document.querySelectorAll('.nav-item.admin-only');
        
        if (role !== 'ADMIN') {
            adminNavItems.forEach(item => {
                item.style.display = 'none';
            });
        }
    }
};

// ========================================
// Logout Handler
// ========================================
const logoutHandler = {
    init() {
        const btn = document.getElementById('logout-btn');
        btn.addEventListener('click', () => {
            auth.logout();
            
            const loginPage = document.getElementById('login-page');
            const mainApp = document.getElementById('main-app');
            
            mainApp.classList.add('hidden');
            loginPage.classList.remove('hidden');
            
            toast.info('Logged out successfully');
        });
    }
};

// ========================================
// Data Manager (Demo Data)
// ========================================
const dataManager = {
    // Demo cases data
    cases: [
        { id: 1, case_no: 'CV/2024/001', title: 'Smith vs. Johnson', type: 'Civil', status: 'ACTIVE', priority: 'HIGH', filed_date: '2024-01-15', assigned_to: 'Judge Williams' },
        { id: 2, case_no: 'CV/2024/002', title: 'Williams vs. Davis', type: 'Civil', status: 'RESOLVED', priority: 'MEDIUM', filed_date: '2024-01-14', assigned_to: 'Judge Brown' },
        { id: 3, case_no: 'CR/2024/003', title: 'State vs. Brown', type: 'Criminal', status: 'ACTIVE', priority: 'CRITICAL', filed_date: '2024-01-13', assigned_to: 'Judge Miller' },
        { id: 4, case_no: 'CV/2024/004', title: 'Anderson vs. Taylor', type: 'Civil', status: 'PENDING', priority: 'HIGH', filed_date: '2024-01-12', assigned_to: 'Judge Williams' },
        { id: 5, case_no: 'FM/2024/001', title: 'Johnson Divorce', type: 'Family', status: 'ACTIVE', priority: 'MEDIUM', filed_date: '2024-01-11', assigned_to: 'Judge Davis' }
    ],

    // Demo documents data
    documents: [
        { id: 1, name: 'complaint.pdf', case_no: 'CV/2024/001', type: 'Complaint', size: '2.4 MB', uploaded: '2024-01-15', status: 'Verified' },
        { id: 2, name: 'petition.docx', case_no: 'CV/2024/002', type: 'Petition', size: '1.2 MB', uploaded: '2024-01-14', status: 'Verified' },
        { id: 3, name: 'evidence.pdf', case_no: 'CR/2024/003', type: 'Evidence', size: '5.6 MB', uploaded: '2024-01-13', status: 'Pending' }
    ],

    // Demo hearings data
    hearings: [
        { id: 1, case_no: 'CV/2024/001', datetime: '2024-01-20 10:00 AM', court: 'Court A', judge: 'Judge Williams', status: 'Scheduled' },
        { id: 2, case_no: 'CR/2024/003', datetime: '2024-01-20 02:00 PM', court: 'Court B', judge: 'Judge Miller', status: 'Scheduled' },
        { id: 3, case_no: 'CV/2024/004', datetime: '2024-01-21 11:00 AM', court: 'Court A', judge: 'Judge Williams', status: 'Scheduled' }
    ],

    // Demo users data
    users: [
        { id: 1, name: 'Admin User', email: 'admin@justiflow.com', role: 'ADMIN', status: 'Active', created: '2024-01-01' },
        { id: 2, name: 'Judge Williams', email: 'judge@justiflow.com', role: 'JUDGE', status: 'Active', created: '2024-01-02' },
        { id: 3, name: 'Clerk Johnson', email: 'clerk@justiflow.com', role: 'CLERK', status: 'Active', created: '2024-01-03' }
    ],

    // Render functions
    renderCasesTable() {
        const tbody = document.getElementById('cases-table');
        if (!tbody) return;

        tbody.innerHTML = this.cases.map(c => `
            <tr>
                <td>${c.case_no}</td>
                <td>${c.title}</td>
                <td>${c.type}</td>
                <td><span class="badge badge-${this.getStatusClass(c.status)}">${c.status}</span></td>
                <td><span class="badge badge-${this.getPriorityClass(c.priority)}">${c.priority}</span></td>
                <td>${c.filed_date}</td>
                <td>${c.assigned_to}</td>
                <td>
                    <button class="btn-icon" title="View"><i class="fas fa-eye"></i></button>
                    <button class="btn-icon" title="Edit"><i class="fas fa-edit"></i></button>
                    <button class="btn-icon" title="Delete"><i class="fas fa-trash"></i></button>
                </td>
            </tr>
        `).join('');
    },

    renderDocumentsTable() {
        const tbody = document.getElementById('documents-table');
        if (!tbody) return;

        tbody.innerHTML = this.documents.map(d => `
            <tr>
                <td><i class="fas fa-file-${this.getFileIcon(d.name)} text-${this.getFileColor(d.name)}"></i> ${d.name}</td>
                <td>${d.case_no}</td>
                <td>${d.type}</td>
                <td>${d.size}</td>
                <td>${d.uploaded}</td>
                <td><span class="badge badge-${d.status === 'Verified' ? 'success' : 'warning'}">${d.status}</span></td>
                <td>
                    <button class="btn-icon" title="View"><i class="fas fa-eye"></i></button>
                    <button class="btn-icon" title="Download"><i class="fas fa-download"></i></button>
                    <button class="btn-icon" title="Delete"><i class="fas fa-trash"></i></button>
                </td>
            </tr>
        `).join('');
    },

    renderHearingsTable() {
        const tbody = document.getElementById('hearings-table');
        if (!tbody) return;

        tbody.innerHTML = this.hearings.map(h => `
            <tr>
                <td>${h.case_no}</td>
                <td>${h.datetime}</td>
                <td>${h.court}</td>
                <td>${h.judge}</td>
                <td><span class="badge badge-primary">${h.status}</span></td>
                <td>
                    <button class="btn-icon" title="View"><i class="fas fa-eye"></i></button>
                    <button class="btn-icon" title="Edit"><i class="fas fa-edit"></i></button>
                </td>
            </tr>
        `).join('');
    },

    renderUsersTable() {
        const tbody = document.getElementById('users-table');
        if (!tbody) return;

        tbody.innerHTML = this.users.map(u => `
            <tr>
                <td>
                    <div class="user-cell">
                        <div class="user-avatar-sm">${u.name.charAt(0)}</div>
                        <span>${u.name}</span>
                    </div>
                </td>
                <td>${u.email}</td>
                <td><span class="badge badge-${u.role === 'ADMIN' ? 'primary' : 'secondary'}">${u.role}</span></td>
                <td><span class="badge badge-success">${u.status}</span></td>
                <td>${u.created}</td>
                <td>
                    <button class="btn-icon" title="Edit"><i class="fas fa-edit"></i></button>
                    <button class="btn-icon" title="Delete"><i class="fas fa-trash"></i></button>
                </td>
            </tr>
        `).join('');
    },

    getStatusClass(status) {
        const classes = {
            'ACTIVE': 'primary',
            'PENDING': 'warning',
            'RESOLVED': 'success',
            'CLOSED': 'secondary'
        };
        return classes[status] || 'secondary';
    },

    getPriorityClass(priority) {
        const classes = {
            'CRITICAL': 'danger',
            'HIGH': 'warning',
            'MEDIUM': 'secondary',
            'LOW': 'success'
        };
        return classes[priority] || 'secondary';
    },

    getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const icons = {
            'pdf': 'pdf',
            'doc': 'word',
            'docx': 'word',
            'xls': 'excel',
            'xlsx': 'excel',
            'ppt': 'powerpoint',
            'pptx': 'powerpoint',
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image',
            'gif': 'image'
        };
        return icons[ext] || 'alt';
    },

    getFileColor(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const colors = {
            'pdf': 'danger',
            'doc': 'primary',
            'docx': 'primary',
            'xls': 'success',
            'xlsx': 'success',
            'ppt': 'warning',
            'pptx': 'warning'
        };
        return colors[ext] || 'gray';
    }
};

// ========================================
// App Initialization
// ========================================
const app = {
    async init() {
        console.log('Initializing JustiFlow DCM System...');
        
        // Initialize all modules
        toast.init();
        modal.init();
        router.init();
        sidebar.init();
        dropdowns.init();
        loginHandler.init();
        logoutHandler.init();
        
        // Check if user is already logged in
        if (auth.checkAuth()) {
            const user = auth.getUser();
            loginHandler.showMainApp(user);
        }

        // Render demo data
        dataManager.renderCasesTable();
        dataManager.renderDocumentsTable();
        dataManager.renderHearingsTable();
        dataManager.renderUsersTable();

        console.log('JustiFlow initialized successfully!');
    }
};

// ========================================
// Start the App
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    app.init();
});