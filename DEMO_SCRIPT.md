# 3-Minute Demo Video Script

## 0:00-0:30 Project Overview

"This is Support Integrity Auditor, an NLP system that checks whether a support ticket's assigned priority matches its objective severity. It uses self-supervised pseudo-labeling because the dataset does not include mismatch labels."

Show:
- `README.md`
- Architecture diagram
- Deliverables list

## 0:30-1:15 Pseudo-Label Strategy

"The system uses two independent severity signals: text urgency and resolution time. Text gets a higher weight because the actual issue description is the strongest evidence. Resolution time adds an operational signal. The fused severity is compared against the assigned priority to create a mismatch label."

Show:
- `sia_core.py`
- `add_pseudo_labels`
- `text_severity_score`
- `resolution_severity_score`

## 1:15-2:00 Hidden Crisis Walkthrough

Use this single-ticket example in Streamlit:

- Subject: `Checkout payments failing`
- Description: `Production checkout is down and customers cannot complete payments.`
- Priority: `Low`
- Channel: `chat`
- Resolution Time: `36`
- Type: `Billing`

Explain:
"The assigned priority is Low, but the inferred severity is higher because the ticket mentions production impact and failed payments. This is a Hidden Crisis."

## 2:00-2:35 False Alarm Walkthrough

Use this single-ticket example:

- Subject: `Typo in dashboard label`
- Description: `There is a minor spelling typo on the settings page.`
- Priority: `Critical`
- Channel: `email`
- Resolution Time: `1`
- Type: `UI`

Explain:
"The assigned priority is Critical, but the actual evidence is minor and resolved quickly. This is a False Alarm."

## 2:35-3:00 Adversarial Demo

Use:

- Subject: `Not urgent but all users blocked`
- Description: `Not urgent is written here, but the production workspace is down and all users are blocked.`
- Priority: `Low`
- Channel: `email`
- Resolution Time: `50`
- Type: `Access`

Explain:
"This ticket tries to fool keyword-only systems with the phrase 'not urgent', but the system still sees production down, blocked users, and long resolution time."

End with:
"The dashboard shows flagged ticket distribution, mismatch types, contributing signal scores, and severity delta by ticket type and channel."
