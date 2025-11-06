export const CSV_TEMPLATES = {
  'Target Lists': {
    headers: ['hcp_id', 'hcp_name', 'specialty', 'territory', 'tier'],
    sampleData: [
      ['HCP001', 'Dr. Rajesh Kumar', 'Cardiology', 'North Delhi', 'A'],
      ['HCP002', 'Dr. Priya Sharma', 'Neurology', 'South Mumbai', 'A'],
      ['HCP003', 'Dr. Amit Patel', 'Orthopedics', 'East Bangalore', 'B'],
      ['HCP004', 'Dr. Sneha Reddy', 'Oncology', 'West Delhi', 'A'],
      ['HCP005', 'Dr. Vikram Singh', 'Gastroenterology', 'Central Gurugram', 'B'],
    ],
    filename: 'target_lists_sample.csv'
  },
  'Call Lists': {
    headers: ['hcp_id', 'hcp_name', 'call_date', 'sales_rep', 'status'],
    sampleData: [
      ['HCP001', 'Dr. Rajesh Kumar', '2025-11-01', 'Arjun Kapoor', 'Scheduled'],
      ['HCP002', 'Dr. Priya Sharma', '2025-11-02', 'Pooja Singh', 'Scheduled'],
      ['HCP003', 'Dr. Amit Patel', '2025-11-03', 'Ravi Verma', 'Pending'],
      ['HCP004', 'Dr. Sneha Reddy', '2025-11-04', 'Neha Agarwal', 'Scheduled'],
      ['HCP005', 'Dr. Vikram Singh', '2025-11-05', 'Rahul Joshi', 'Completed'],
    ],
    filename: 'call_lists_sample.csv'
  },
  'Formulary Decision-Maker Lists': {
    headers: ['contact_id', 'contact_name', 'organization', 'email', 'influence_level'],
    sampleData: [
      ['FDM001', 'Dr. Rajiv Malhotra', 'Max Healthcare', 'rajiv.malhotra@max.com', 'High'],
      ['FDM002', 'Dr. Sunita Verma', 'Apollo Hospitals', 'sunita.verma@apollo.com', 'High'],
      ['FDM003', 'Mr. Anil Chopra', 'Fortis Healthcare', 'anil.chopra@fortis.com', 'Medium'],
      ['FDM004', 'Dr. Kiran Patel', 'Medanta Hospital', 'kiran.patel@medanta.com', 'High'],
      ['FDM005', 'Ms. Rekha Nair', 'AIIMS Delhi', 'rekha.nair@aiims.edu', 'High'],
    ],
    filename: 'formulary_decision_maker_lists_sample.csv'
  },
  'IDN/Health System Lists': {
    headers: ['system_id', 'system_name', 'contact_name', 'contact_email', 'importance'],
    sampleData: [
      ['IDN001', 'Apollo Hospitals Group', 'Dr. Ravi Shankar', 'ravi.shankar@apollo.com', 'Tier 1'],
      ['IDN002', 'Fortis Healthcare Network', 'Ms. Priya Mathur', 'priya.mathur@fortis.com', 'Tier 1'],
      ['IDN003', 'Max Healthcare System', 'Mr. Sandeep Gupta', 'sandeep.gupta@max.com', 'Tier 2'],
      ['IDN004', 'Medanta Network', 'Dr. Arvind Kumar', 'arvind.kumar@medanta.com', 'Tier 1'],
      ['IDN005', 'Narayana Health Group', 'Dr. Lakshmi Rao', 'lakshmi.rao@narayana.com', 'Tier 2'],
    ],
    filename: 'idn_health_system_lists_sample.csv'
  },
  'Event Invitation Lists': {
    headers: ['event_name', 'event_date', 'invitee_id', 'invitee_name', 'email', 'status'],
    sampleData: [
      ['Cardiology Summit 2025', '2025-12-15', 'HCP001', 'Dr. Rajesh Kumar', 'rajesh.kumar@apollo.com', 'Invited'],
      ['Neurology Conference', '2025-12-20', 'HCP002', 'Dr. Priya Sharma', 'priya.sharma@fortis.com', 'Confirmed'],
      ['Oncology Workshop', '2026-01-10', 'HCP004', 'Dr. Sneha Reddy', 'sneha.reddy@aiims.edu', 'Invited'],
      ['Gastro Symposium', '2026-01-25', 'HCP005', 'Dr. Vikram Singh', 'vikram.singh@medanta.com', 'Pending'],
      ['Endocrine Conclave', '2026-02-05', 'HCP006', 'Dr. Ananya Iyer', 'ananya.iyer@apollo.com', 'Confirmed'],
    ],
    filename: 'event_invitation_lists_sample.csv'
  },
  'Digital Engagement Lists': {
    headers: ['contact_id', 'contact_name', 'email', 'specialty', 'opt_in'],
    sampleData: [
      ['DG001', 'Dr. Rajesh Kumar', 'rajesh.kumar@apollo.com', 'Cardiology', 'TRUE'],
      ['DG002', 'Dr. Priya Sharma', 'priya.sharma@fortis.com', 'Neurology', 'TRUE'],
      ['DG003', 'Dr. Amit Patel', 'amit.patel@max.com', 'Orthopedics', 'FALSE'],
      ['DG004', 'Dr. Sneha Reddy', 'sneha.reddy@aiims.edu', 'Oncology', 'TRUE'],
      ['DG005', 'Dr. Vikram Singh', 'vikram.singh@medanta.com', 'Gastroenterology', 'TRUE'],
    ],
    filename: 'digital_engagement_lists_sample.csv'
  },
  'High-Value Prescriber Lists': {
    headers: ['hcp_id', 'hcp_name', 'specialty', 'territory', 'total_prescriptions', 'revenue', 'value_tier'],
    sampleData: [
      ['HCP001', 'Dr. Rajesh Kumar', 'Cardiology', 'North Delhi', '850', '125000.50', 'Platinum'],
      ['HCP002', 'Dr. Priya Sharma', 'Neurology', 'South Mumbai', '720', '98000.75', 'Gold'],
      ['HCP003', 'Dr. Amit Patel', 'Orthopedics', 'East Bangalore', '550', '72000.00', 'Gold'],
      ['HCP004', 'Dr. Sneha Reddy', 'Oncology', 'West Delhi', '920', '145000.25', 'Platinum'],
      ['HCP005', 'Dr. Vikram Singh', 'Gastroenterology', 'Central Gurugram', '480', '65000.00', 'Silver'],
    ],
    filename: 'high_value_prescriber_lists_sample.csv'
  },
  'Competitor Target Lists': {
    headers: ['hcp_id', 'hcp_name', 'specialty', 'territory', 'competitor_product', 'conversion_potential', 'assigned_rep'],
    sampleData: [
      ['HCP011', 'Dr. Ramesh Patel', 'Cardiology', 'North Delhi', 'CompetitorX CardioMed', 'High', 'Arjun Kapoor'],
      ['HCP012', 'Dr. Lakshmi Menon', 'Neurology', 'South Mumbai', 'CompetitorY NeuroPlus', 'Medium', 'Pooja Singh'],
      ['HCP013', 'Dr. Sunil Reddy', 'Orthopedics', 'East Bangalore', 'CompetitorZ BoneCare', 'High', 'Ravi Verma'],
      ['HCP014', 'Dr. Anjali Desai', 'Oncology', 'West Delhi', 'CompetitorX OncoMax', 'High', 'Neha Agarwal'],
      ['HCP015', 'Dr. Karthik Iyer', 'Gastroenterology', 'Central Gurugram', 'CompetitorY GastroRelief', 'Low', 'Rahul Joshi'],
    ],
    filename: 'competitor_target_lists_sample.csv'
  },
};

export function arrayToCSV(headers: string[], data: string[][]): string {
  const rows = [headers, ...data];
  return rows.map(row => row.join(',')).join('\n');
}

export function downloadCSV(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  URL.revokeObjectURL(url);
}

export function downloadSampleCSV(listType: string): void {
  const template = CSV_TEMPLATES[listType as keyof typeof CSV_TEMPLATES];
  
  if (!template) {
    console.error(`No template found for list type: ${listType}`);
    return;
  }
  
  const csvContent = arrayToCSV(template.headers, template.sampleData);
  downloadCSV(csvContent, template.filename);
}

export function getListTypeFromSubdomain(subdomainName: string): string | null {
  const mapping: Record<string, string> = {
    'Target Lists': 'Target Lists',
    'Call Lists': 'Call Lists',
    'Formulary Decision-Maker Lists': 'Formulary Decision-Maker Lists',
    'IDN/Health System Lists': 'IDN/Health System Lists',
    'Event Invitation Lists': 'Event Invitation Lists',
    'Digital Engagement Lists': 'Digital Engagement Lists',
    'High-Value Prescriber Lists': 'High-Value Prescriber Lists',
    'Competitor Target Lists': 'Competitor Target Lists',
  };
  
  return mapping[subdomainName] || null;
}
