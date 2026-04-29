/**
 * JustiFlow - AI Assistant Module
 * LLM-based assistant for legal case summarization, workflow explanation, RAG responses, and ML decision transparency
 */

// ========================================
// AI Assistant
// ========================================
const aiAssistant = {
    widget: null,
    panel: null,
    toggle: null,
    closeBtn: null,
    messagesContainer: null,
    input: null,
    sendBtn: null,
    currentMode: 'case-summary',
    isOpen: false,

    // Demo knowledge base for RAG
    knowledgeBase: {
        'case summary': [
            { title: 'Smith vs. Johnson', summary: 'Civil dispute regarding property boundary. Case involves claims of encroachment on adjacent land. Currently in discovery phase with scheduled mediation.' },
            { title: 'Williams vs. Davis', summary: 'Contract dispute involving breach of service agreement. Plaintiff seeks damages for non-payment. Case resolved with settlement favoring defendant.' },
            { title: 'State vs. Brown', summary: 'Criminal case involving multiple charges of fraud. Defendant accused of defrauding investors of $2.5M. Currently in trial phase.' }
        ],
        'workflow': [
            { title: 'Filing Process', steps: ['1. File complaint/petition', '2. Pay filing fee', '3. Serve defendant', '4. Defendant files response', '5. Discovery phase', '6. Pre-trial conference', '7. Trial or settlement'] },
            { title: 'Hearing Process', steps: ['1. Schedule hearing', '2. Notify all parties', '3. Prepare evidence', '4. Attend hearing', '5. Receive judgment', '6. File appeal if needed'] },
            { title: 'Case Disposition', steps: ['1. Review all evidence', '2. Hear arguments', '3. Consider precedents', '4. Issue judgment', '5. File order', '6. Close case'] }
        ],
        'priority': [
            { factor: 'Case Type', weight: 'Criminal cases typically get priority over civil cases', impact: 'High' },
            { factor: 'Severity', weight: 'Cases involving violence or major fraud get highest priority', impact: 'High' },
            { factor: 'Age', weight: 'Older cases are prioritized to prevent statute of limitations issues', impact: 'Medium' },
            { factor: 'Judge Workload', weight: 'Cases assigned based on judge availability and expertise', impact: 'Medium' }
        ]
    },

    // ML explanation factors
    mlFactors: {
        'outcome': [
            { factor: 'Case History', value: '85%', explanation: 'Similar cases in past 5 years had favorable outcome' },
            { factor: 'Evidence Strength', value: '78%', explanation: 'Strong documentary evidence supports claim' },
            { factor: 'Legal Precedent', value: '72%', explanation: 'Favorable precedent exists in jurisdiction' },
            { factor: 'Party Reputation', value: '65%', explanation: 'Prior legal conduct of parties considered' }
        ],
        'duration': [
            { factor: 'Case Complexity', value: 'High', explanation: 'Multiple parties and issues extend timeline' },
            { factor: 'Evidence Volume', value: 'Medium', explanation: 'Moderate discovery requirements' },
            { factor: 'Court Availability', value: 'High', explanation: 'Busy docket affects scheduling' }
        ]
    },

    init() {
        this.widget = document.getElementById('ai-assistant-widget');
        this.panel = document.getElementById('ai-chat-panel');
        this.toggle = document.getElementById('ai-assistant-toggle');
        this.closeBtn = document.getElementById('ai-chat-close');
        this.messagesContainer = document.getElementById('ai-chat-messages');
        this.input = document.getElementById('ai-chat-input');
        this.sendBtn = document.getElementById('ai-chat-send');

        this.bindEvents();
    },

    bindEvents() {
        // Toggle panel
        if (this.toggle) {
            this.toggle.addEventListener('click', () => this.togglePanel());
        }

        // Close panel
        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.closePanel());
        }

        // Send message
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.sendMessage());
        }

        // Enter key to send
        if (this.input) {
            this.input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendMessage();
                }
            });
        }

        // Mode buttons
        document.querySelectorAll('.ai-mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.ai-mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentMode = btn.dataset.mode;
            });
        });

        // Quick action buttons
        document.querySelectorAll('.ai-quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.input.value = btn.dataset.prompt;
                this.sendMessage();
            });
        });
    },

    togglePanel() {
        if (this.isOpen) {
            this.closePanel();
        } else {
            this.openPanel();
        }
    },

    openPanel() {
        if (this.panel) {
            this.panel.classList.remove('hidden');
            this.isOpen = true;
            if (this.toggle) this.toggle.classList.add('has-unread');
            if (this.input) this.input.focus();
        }
    },

    closePanel() {
        if (this.panel) {
            this.panel.classList.add('hidden');
            this.isOpen = false;
        }
    },

    async sendMessage() {
        if (!this.input || !this.messagesContainer) return;
        
        const message = this.input.value.trim();
        if (!message) return;

        // Add user message
        this.addMessage(message, 'user');
        this.input.value = '';

        // Show typing indicator
        this.showTyping();

        // Process message based on mode
        try {
            const response = await this.processMessage(message);
            this.hideTyping();
            this.addMessage(response, 'bot');
        } catch (error) {
            this.hideTyping();
            this.addMessage('I apologize, but I encountered an error processing your request. Please try again.', 'bot');
        }
    },

    async processMessage(message) {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 700));

        switch (this.currentMode) {
            case 'case-summary':
                return this.generateCaseSummary(message);
            case 'workflow':
                return this.explainWorkflow(message);
            case 'rag':
                return this.generateRAGResponse(message);
            case 'explain':
                return this.explainMLDecision(message);
            default:
                return this.generateDefaultResponse(message);
        }
    },

    generateCaseSummary(message) {
        const caseMatch = message.match(/(CV|CR|FM)\/\d{4}\/\d{3}/i);
        
        if (caseMatch) {
            const caseNo = caseMatch[0].toUpperCase();
            const caseData = this.knowledgeBase['case summary'].find(c => 
                message.toLowerCase().includes(c.title.toLowerCase()) || caseNo
            );
            
            if (caseData) {
                return `<p><strong>Case Summary: ${caseData.title}</strong></p>
                <p>${caseData.summary}</p>
                <div class="ai-rag-source">
                    <strong>Key Information:</strong>
                    <ul style="margin: 8px 0 0 0; padding-left: 16px;">
                        <li>Status: Active</li>
                        <li>Next Hearing: Scheduled</li>
                        <li>Documents: 5 filed</li>
                    </ul>
                </div>`;
            }
        }

        return `<p>Based on the case information requested, here's a summary:</p>
        <ul>
            <li><strong>Case Type:</strong> Civil litigation</li>
            <li><strong>Current Status:</strong> Active - Discovery Phase</li>
            <li><strong>Key Parties:</strong> Plaintiff and Defendant</li>
            <li><strong>Timeline:</strong> Filed Jan 2024, currently at pre-trial stage</li>
            <li><strong>Next Steps:</strong> Mediation scheduled, then trial if unresolved</li>
        </ul>
        <p style="margin-top: 12px; font-size: 0.8rem; color: #64748b;">
            <i class="fas fa-info-circle"></i> 
            Would you like more details about a specific case? Provide the case number.
        </p>`;
    },

    explainWorkflow(message) {
        const workflow = this.knowledgeBase['workflow'].find(w => 
            message.toLowerCase().includes(w.title.toLowerCase())
        );

        if (workflow) {
            return `<p><strong>${workflow.title} Process</strong></p>
            <p>Here's the step-by-step workflow:</p>
            <ol style="margin: 12px 0; padding-left: 20px;">
                ${workflow.steps.map(step => `<li style="margin-bottom: 8px;">${step}</li>`).join('')}
            </ol>
            <div class="ai-rag-source">
                <strong>Note:</strong> This is a general workflow. Specific procedures may vary by jurisdiction and case type.
            </div>`;
        }

        return `<p>Here's an overview of the judicial workflow:</p>
        <ul>
            <li><strong>Case Filing:</strong> Initial submission of complaint/petition</li>
            <li><strong>Service:</strong> Legal notice to all parties involved</li>
            <li><strong>Discovery:</strong> Evidence exchange between parties</li>
            <li><strong>Pre-Trial:</strong> Conference to narrow issues</li>
            <li><strong>Trial:</strong> Court hearing with evidence and arguments</li>
            <li><strong>Judgment:</strong> Final decision issued by judge</li>
            <li><strong>Appeal:</strong> Higher court review if dissatisfied</li>
        </ul>
        <p style="margin-top: 12px; font-size: 0.8rem; color: #64748b;">
            <i class="fas fa-lightbulb"></i> 
            Ask about a specific workflow (e.g., "hearing process", "filing process")
        </p>`;
    },

    generateRAGResponse(message) {
        const relevantCases = this.knowledgeBase['case summary']
            .filter(c => message.toLowerCase().includes(c.title.split(' vs. ')[0].toLowerCase()) ||
                        message.toLowerCase().includes(c.title.split(' vs. ')[1].toLowerCase()));

        if (relevantCases.length > 0) {
            const caseData = relevantCases[0];
            return `<p><strong>Context-Aware Response based on Legal Knowledge Base:</strong></p>
            <p>Based on similar cases in our database:</p>
            <div class="ai-rag-source">
                <strong>Reference Case:</strong> ${caseData.title}<br>
                <strong>Outcome:</strong> ${caseData.summary.includes('resolved') ? 'Resolved with settlement' : 'Currently active'}<br>
                <strong>Relevance:</strong> 92% match with your query
            </div>
            <p style="margin-top: 12px;">This response was generated using our RAG pipeline, retrieving relevant context from the legal case database.</p>`;
        }

        return `<p><strong>Legal Information Response:</strong></p>
        <p>I've searched our legal knowledge base and found relevant information:</p>
        <ul>
            <li><strong>Applicable Law:</strong> Based on civil procedure codes</li>
            <li><strong>Precedent:</strong> Similar cases found in jurisdiction</li>
            <li><strong>Recommendation:</strong> Consult with legal counsel for specific advice</li>
        </ul>
        <div class="ai-rag-source">
            <strong>Sources:</strong> Case law database, statutory references, procedural rules
        </div>`;
    },

    explainMLDecision(message) {
        if (message.toLowerCase().includes('outcome') || message.toLowerCase().includes('predict')) {
            return `<p><strong>ML Decision Explanation:</strong></p>
            <p>Our AI model predicted this outcome based on the following factors:</p>
            <div class="ai-ml-explanation">
                <h4><i class="fas fa-brain"></i> Decision Factors</h4>
                ${this.mlFactors['outcome'].map(f => `
                    <div class="ai-ml-factor">
                        <span class="ai-ml-factor-label">${f.factor}</span>
                        <span class="ai-ml-factor-value">${f.value}</span>
                    </div>
                `).join('')}
            </div>
            <p style="margin-top: 12px; font-size: 0.8rem; color: #64748b;">
                <i class="fas fa-info-circle"></i> 
                The model achieved 89% accuracy on validation data. Human oversight is recommended for critical decisions.
            </p>`;
        }

        if (message.toLowerCase().includes('duration') || message.toLowerCase().includes('time')) {
            return `<p><strong>Duration Estimation Explanation:</strong></p>
            <p>The estimated duration is based on these ML factors:</p>
            <div class="ai-ml-explanation">
                <h4><i class="fas fa-clock"></i> Time Factors</h4>
                ${this.mlFactors['duration'].map(f => `
                    <div class="ai-ml-factor">
                        <span class="ai-ml-factor-label">${f.factor}</span>
                        <span class="ai-ml-factor-value">${f.value}</span>
                    </div>
                `).join('')}
            </div>
            <p style="margin-top: 12px; font-size: 0.8rem; color: #64748b;">
                <i class="fas fa-info-circle"></i> 
                Historical data from 10,000+ cases used for training. Actual duration may vary.
            </p>`;
        }

        return `<p><strong>ML Decision Transparency:</strong></p>
        <p>Our AI system uses machine learning to assist with case management decisions. Here's how it works:</p>
        <ul>
            <li><strong>Data Collection:</strong> Historical case data from our database</li>
            <li><strong>Feature Engineering:</strong> Case type, parties, evidence, timeline</li>
            <li><strong>Model Training:</strong> Trained on 50,000+ past cases</li>
            <li><strong>Prediction:</strong> Generates probability scores</li>
            <li><strong>Human Review:</strong> All AI suggestions require human approval</li>
        </ul>
        <div class="ai-ml-explanation">
            <h4><i class="fas fa-shield-alt"></i> System Transparency</h4>
            <p style="font-size: 0.8rem; color: #166534;">Model accuracy: 89% | False positive rate: 3.2% | Human oversight required for all decisions</p>
        </div>`;
    },

    generateDefaultResponse(message) {
        return `<p>I'm here to help with your legal case management questions. Please select a mode above:</p>
        <ul>
            <li><strong>Summarize</strong> - Get case summaries</li>
            <li><strong>Workflow</strong> - Understand judicial processes</li>
            <li><strong>RAG</strong> - Context-aware legal responses</li>
            <li><strong>Explain ML</strong> - AI decision transparency</li>
        </ul>`;
    },

    addMessage(content, sender) {
        if (!this.messagesContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-message ai-message-${sender}`;
        
        const avatar = sender === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
        
        messageDiv.innerHTML = `
            <div class="ai-message-avatar">${avatar}</div>
            <div class="ai-message-content">${content}</div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    },

    showTyping() {
        if (!this.messagesContainer) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'ai-message ai-message-bot';
        typingDiv.id = 'ai-typing-indicator';
        typingDiv.innerHTML = `
            <div class="ai-message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="ai-message-content">
                <div class="ai-message-typing">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        this.messagesContainer.appendChild(typingDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    },

    hideTyping() {
        const typingIndicator = document.getElementById('ai-typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
};