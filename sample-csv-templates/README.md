# Sample CSV Templates for PharmaDB List Management

This directory contains standardized CSV templates for all list types in the PharmaDB system. Each template follows a predefined format that matches the database schema to ensure data consistency and prevent synchronization issues.

## 📋 Available Templates

### 🏥 Customer/HCP Lists

#### 1. Target Lists (`target_lists.csv`)
**Purpose**: Track healthcare professionals targeted for outreach  
**Required Fields**:
- `hcp_id` (Required): Unique identifier for the healthcare professional
- `hcp_name` (Optional): Full name of the HCP
- `specialty` (Optional): Medical specialty (e.g., Cardiology, Neurology)
- `territory` (Optional): Geographic territory assignment
- `tier` (Optional): Priority tier (A/B/C)

**Example**:
```csv
hcp_id,hcp_name,specialty,territory,tier
HCP001,Dr. Rajesh Kumar,Cardiology,North Delhi,A
```

#### 2. Call Lists (`call_lists.csv`)
**Purpose**: Schedule and track sales representative calls  
**Required Fields**:
- `hcp_id` (Required): HCP unique identifier
- `hcp_name` (Optional): HCP full name
- `call_date` (Required): Scheduled call date (YYYY-MM-DD format)
- `sales_rep` (Optional): Assigned sales representative name
- `status` (Optional): Call status (Scheduled/Pending/Completed)

**Example**:
```csv
hcp_id,hcp_name,call_date,sales_rep,status
HCP001,Dr. Rajesh Kumar,2025-11-01,Arjun Kapoor,Scheduled
```

---

### 🏢 Account/Institutional Lists

#### 3. Formulary Decision-Maker Lists (`formulary_decision_maker_lists.csv`)
**Purpose**: Track key decision-makers for formulary inclusion  
**Required Fields**:
- `contact_id` (Required): Unique identifier for the contact
- `contact_name` (Optional): Contact's full name
- `organization` (Required): Hospital/healthcare organization name
- `email` (Optional): Contact email address
- `influence_level` (Optional): Level of influence (Low/Medium/High/Very High)

**Example**:
```csv
contact_id,contact_name,organization,email,influence_level
FDM001,Dr. Rajiv Malhotra,Max Healthcare,rajiv.malhotra@max.com,High
```

#### 4. IDN/Health System Lists (`idn_health_system_lists.csv`)
**Purpose**: Manage integrated delivery networks and health systems  
**Required Fields**:
- `system_id` (Required): Unique identifier for the health system
- `system_name` (Required): Name of the IDN/health system
- `contact_name` (Optional): Primary contact person
- `contact_email` (Optional): Contact email address
- `importance` (Optional): Strategic importance (Low/Medium/High/Critical)

**Example**:
```csv
system_id,system_name,contact_name,contact_email,importance
IDN001,Apollo Hospitals Group,Dr. Ravi Shankar,ravi.shankar@apollo.com,Critical
```

---

### 📢 Marketing Campaign Lists

#### 5. Event Invitation Lists (`event_invitation_lists.csv`)
**Purpose**: Manage invitations for medical events and conferences  
**Required Fields**:
- `event_name` (Required): Name of the event
- `event_date` (Required): Event date (YYYY-MM-DD format)
- `invitee_id` (Required): Unique identifier for the invitee
- `invitee_name` (Optional): Invitee's full name
- `email` (Required): Invitee email address
- `status` (Optional): Invitation status (Invited/Confirmed/Declined/Pending)

**Example**:
```csv
event_name,event_date,invitee_id,invitee_name,email,status
Cardiology Summit 2025,2025-12-15,HCP001,Dr. Rajesh Kumar,rajesh.kumar@apollo.com,Invited
```

#### 6. Digital Engagement Lists (`digital_engagement_lists.csv`)
**Purpose**: Manage digital marketing and email campaign recipients  
**Required Fields**:
- `contact_id` (Required): Unique contact identifier
- `contact_name` (Optional): Contact's full name
- `email` (Required): Email address for digital engagement
- `specialty` (Optional): Medical specialty
- `opt_in` (Optional): Email opt-in status (TRUE/FALSE)

**Example**:
```csv
contact_id,contact_name,email,specialty,opt_in
DG001,Dr. Rajesh Kumar,rajesh.kumar@apollo.com,Cardiology,TRUE
```

---

### Data/Analytics Lists

#### 7. High-Value Prescriber Lists (`high_value_prescriber_lists.csv`)
**Purpose**: Track high-revenue generating prescribers  
**Required Fields**:
- `hcp_id` (Required): HCP unique identifier
- `hcp_name` (Optional): HCP full name
- `specialty` (Optional): Medical specialty
- `territory` (Optional): Territory assignment
- `total_prescriptions` (Optional): Total prescription count (integer)
- `revenue` (Optional): Revenue generated (decimal number)
- `value_tier` (Optional): Value classification (Platinum/Gold/Silver/Bronze)

**Example**:
```csv
hcp_id,hcp_name,specialty,territory,total_prescriptions,revenue,value_tier
HCP001,Dr. Rajesh Kumar,Cardiology,North Delhi,850,125000.50,Platinum
```

#### 8. Competitor Target Lists (`competitor_target_lists.csv`)
**Purpose**: Track HCPs using competitor products for conversion  
**Required Fields**:
- `hcp_id` (Required): HCP unique identifier
- `hcp_name` (Optional): HCP full name
- `specialty` (Optional): Medical specialty
- `territory` (Optional): Territory assignment
- `competitor_product` (Optional): Competitor product name
- `conversion_potential` (Optional): Likelihood of conversion (Low/Medium/High/Very High)
- `assigned_rep` (Optional): Sales rep assigned for conversion

**Example**:
```csv
hcp_id,hcp_name,specialty,territory,competitor_product,conversion_potential,assigned_rep
HCP011,Dr. Ramesh Patel,Cardiology,North Delhi,CompetitorX CardioMed,High,Arjun Kapoor
```

---

## Usage Instructions

### How to Use These Templates

1. **Download the Template**: Choose the appropriate CSV template for your list type
2. **Fill in Your Data**: Replace sample data with your actual data, maintaining the exact column headers
3. **Maintain Format**: Keep the same column order and header names
4. **Date Format**: Use YYYY-MM-DD format for all date fields (e.g., 2025-11-01)
5. **Boolean Values**: Use TRUE/FALSE (all caps) for boolean fields like `opt_in`
6. **Upload**: Use the "Bulk Upload" feature in PharmaDB to import your CSV

### Important Notes

**Critical Requirements**:
- **Do NOT modify column headers** - They must match exactly as shown
- **Do NOT change column order** - Keep columns in the same sequence
- **Required fields must have values** - Fields marked as "Required" cannot be empty
- **Use consistent date format** - Always use YYYY-MM-DD
- **Avoid special characters** - Use standard ASCII characters in text fields
- **Save as UTF-8 CSV** - Ensure proper encoding when saving the file

### Common Mistakes to Avoid

**Don't**:
- Add extra columns not in the template
- Remove required columns
- Use different date formats (e.g., MM/DD/YYYY, DD-MM-YYYY)
- Include column headers in data rows
- Use Excel formulas or formatting

**Do**:
- Keep all column headers exactly as provided
- Leave optional fields empty if you don't have data
- Use consistent naming conventions
- Validate data before upload
- Test with a small batch first

---

## Field Validation Rules

### Common Field Types

| Field Type | Format | Example | Notes |
|------------|--------|---------|-------|
| ID Fields | Alphanumeric | HCP001, FDM005 | No spaces, consistent prefix |
| Dates | YYYY-MM-DD | 2025-11-01 | ISO 8601 format |
| Email | Standard email | user@example.com | Valid email format |
| Boolean | TRUE/FALSE | TRUE | All caps |
| Numbers | Integer/Decimal | 850, 125000.50 | No commas or currency symbols |
| Text | Plain text | Dr. John Doe | Avoid special characters |

### Tier/Level Values

- **Tiers**: A, B, C
- **Influence Levels**: Low, Medium, High, Very High
- **Value Tiers**: Bronze, Silver, Gold, Platinum
- **Status Values**: Scheduled, Pending, Completed, Invited, Confirmed, Declined
- **Importance**: Low, Medium, High, Critical
- **Conversion Potential**: Low, Medium, High, Very High

---

## Quick Start Guide

1. **Identify Your List Type**: Determine which domain and list type you need
2. **Download Template**: Get the corresponding CSV template from this directory
3. **Prepare Your Data**: Organize your data according to the template structure
4. **Fill the Template**: Replace sample data with your actual data
5. **Validate**: Check that all required fields are filled
6. **Upload**: Go to PharmaDB → Select Domain → Create List → Bulk Upload
7. **Verify**: Check that data appears correctly in the system

---

## 📞 Support

If you encounter issues with CSV uploads:
- Verify column headers match exactly
- Check date formats are YYYY-MM-DD
- Ensure required fields are populated
- Validate email addresses are properly formatted
- Confirm numeric fields don't contain text

For technical support, contact your system administrator.

---

## Version Information

**Last Updated**: October 27, 2025  
**Schema Version**: 1.0  
**Compatibility**: PharmaDB v1.0+

---

## Related Documentation

- [Database Schema Reference](../backend/app/schemas/)
- [API Documentation](../backend/README.md)
- [User Guide](../README.md)
- [Integration Guide](../INTEGRATION.md)

