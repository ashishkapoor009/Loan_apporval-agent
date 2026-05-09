import sqlite3

def setup_database():
    conn = sqlite3.connect('loans.db')
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT,
            income REAL,
            loan_amount REAL,
            credit_score INTEGER,
            employment_status TEXT,
            scenario_type TEXT
        )
    ''')
    
    # Clear existing data
    cursor.execute('DELETE FROM loan_applications')
    
    # Sample scenarios
    scenarios = [
        # 1. Clear Approval
        ('Alice Smith', 120000, 20000, 780, 'Employed', 'Clear Approval'),
        # 2. Clear Rejection
        ('Bob Jones', 30000, 50000, 550, 'Unemployed', 'Clear Rejection'),
        # 3. Borderline Case - Manual Review
        ('Charlie Brown', 65000, 30000, 640, 'Employed', 'Borderline Case'),
        # 4. Policy Conflict - Manual Review (High credit but unemployed)
        ('Diana Prince', 80000, 15000, 750, 'Unemployed', 'Policy Conflict'),
        # 5. Policy Conflict / Priority Logic (High Income but very low credit score)
        ('Evan Wright', 200000, 40000, 580, 'Employed', 'Policy Conflict / Priority Logic'),
    ]
    
    cursor.executemany('''
        INSERT INTO loan_applications (applicant_name, income, loan_amount, credit_score, employment_status, scenario_type)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', scenarios)
    
    conn.commit()
    conn.close()
    print("Database 'loans.db' created successfully with 5 sample scenarios.")

if __name__ == '__main__':
    setup_database()
