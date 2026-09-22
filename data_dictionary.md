# Data Dictionary: Indian E-Commerce Customer Behavior & Purchase

## Overview
- **Dataset File**: `data/raw/Ecommerce.csv`
- **Total Records**: 25,000 interactions
- **Total Attributes**: 29 columns
- **Unique Customers**: 8,442
- **Unique Products**: 899
- **Product Categories**: 8
- **Temporal Range**: 2024-01-01 to 2024-12-30 (365 calendar days)
- **Primary Research Task**: *"Given a customer, candidate product, and context available at recommendation time, rank candidate products according to purchase relevance."*
- **Primary Target**: `purchased` (Binary: 1 = Purchased, 0 = Not Purchased)

---

## Column Classification & Modeling Role

| Column Name | Data Type | Description | Role in Primary Recommendation Model | Leakage Status |
| :--- | :--- | :--- | :--- | :--- |
| `customer_id` | `int64` | Unique customer identifier | **Identifier** (cohort tracking, past history linkage) | Safe |
| `session_id` | `int64` | Unique session identifier | **Identifier** (row key) | Safe |
| `product_id` | `int64` | Unique product identifier (899 products) | **Content Feature** (Candidate product entity) | Safe |
| `visit_date` | `object` (`str`)| Date of interaction (`DD-MM-YYYY`) | **Contextual Feature** (Temporal splitting & point-in-time sequencing) | Safe |
| `visit_day` | `int64` | Calendar day of month (1 to 31) | **Contextual Feature** (Recommendation time context) | Safe |
| `visit_month` | `int64` | Calendar month (1 to 12) | **Contextual Feature** (Seasonality context) | Safe |
| `visit_weekday` | `int64` | Day of week (0: Monday to 6: Sunday) | **Contextual Feature** (Day-level context) | Safe |
| `visit_season` | `int64` | Season code (0: Winter, 1: Spring, 2: Summer, 3: Fall) | **Contextual Feature** (Seasonal context) | Safe |
| `device_type` | `int64` | Device used (0: Mobile, 1: Desktop, 2: Tablet) | **Contextual Feature** (Client environment) | Safe |
| `user_type` | `int64` | Customer type (0: New, 1: Returning) | **Contextual Feature** (User lifecycle state) | Safe |
| `marketing_channel` | `int64` | Attribution channel (0 to 5: Direct, Organic, Paid, etc.) | **Contextual Feature** (Traffic source / intent) | Safe |
| `location` | `int64` | Geolocation region code (0 to 224) | **Contextual Feature** (Geographic context) | Safe |
| `product_category` | `int64` | Product category identifier (8 categories: 0 to 7) | **Content Feature** (Candidate product metadata) | Safe |
| `unit_price` | `float64` | Price per unit of candidate product | **Content Feature** (Candidate product metadata) | Safe |
| `discount_percent` | `int64` | Discount percentage on candidate product | **Content Feature** (Candidate product metadata) | Safe |
| `discount_amount` | `float64` | Monetary discount on candidate product | **Content Feature** (Candidate product metadata) | Safe |
| `purchased` | `int64` | Purchase outcome (1 = Purchased, 0 = Not Purchased) | **Primary Target / Ground Truth Label** | Target |
| `added_to_cart` | `int64` | 1 if added to cart in current session, 0 otherwise | **EXCLUDED FROM PREDICTIVE MODEL** (Recommendation occurs pre-cart addition. Retained only for historical feature computation if in past sessions $t_{prev} < t$). | **LEAKAGE-PRONE IF CURRENT** |
| `revenue` | `float64` | Transaction monetary value ($0.0$ if unpurchased) | **EXCLUDED FROM PREDICTIVE MODEL** (Direct outcome of purchase decision. Only used as historical aggregate from past sessions). | **POST-DECISION LEAKAGE** |
| `rating` | `int64` | Post-purchase customer review rating (1 to 5) | **EXCLUDED FROM PREDICTIVE MODEL** (Post-decision feedback). | **POST-DECISION LEAKAGE** |
| `review_text` | `int64` | Review length / type code | **EXCLUDED FROM PREDICTIVE MODEL** (Post-decision feedback). | **POST-DECISION LEAKAGE** |
| `review_helpful_votes`| `int64` | Review votes | **EXCLUDED FROM PREDICTIVE MODEL** (Post-decision feedback). | **POST-DECISION LEAKAGE** |
| `cart_abandoned` | `int64` | Cart abandonment flag | **EXCLUDED FROM PREDICTIVE MODEL** (Post-recommendation funnel outcome). | **POST-DECISION LEAKAGE** |
| `payment_method` | `int64` | Payment method chosen at checkout | **EXCLUDED FROM PREDICTIVE MODEL** (Checkout-stage selection). | **CHECKOUT LEAKAGE** |
| `pages_viewed` | `int64` | Pages viewed in current session | **EXCLUDED FROM PREDICTIVE MODEL** (Full session total. Used only as historical mean from past sessions). | **SESSION LEAKAGE** |
| `time_on_site_sec` | `int64` | Session duration in seconds | **EXCLUDED FROM PREDICTIVE MODEL** (Full session total. Used only as historical mean from past sessions). | **SESSION LEAKAGE** |
| `session_duration_bucket`| `str` | Categorical duration bucket | **EXCLUDED FROM PREDICTIVE MODEL** (Full session total). | **SESSION LEAKAGE** |
| `quantity` | `int64` | Quantity ordered | **EXCLUDED FROM PREDICTIVE MODEL** (Checkout outcome). | **CHECKOUT LEAKAGE** |
| `revenue_normalized`| `float64` | Normalized revenue | **EXCLUDED FROM PREDICTIVE MODEL** (Post-decision). | **POST-DECISION LEAKAGE** |

---

## Point-in-Time Historical Behavioral Features
For each session at timestamp $t$, behavioral features are calculated **strictly from prior customer sessions** ($t_{prev} < t$):
1. `hist_interaction_count`: Number of prior interactions ($N_{hist}$).
2. `hist_purchase_count`: Number of prior purchases.
3. `hist_purchase_rate`: Historical conversion rate ($\text{purchases} / \max(1, N_{hist})$).
4. `hist_revenue`: Total past monetary spend.
5. `hist_cart_add_rate`: Historical add-to-cart rate from prior sessions.
6. `hist_avg_pages_viewed`: Average pages viewed across prior sessions.
7. `hist_avg_time_on_site`: Average dwell time across prior sessions.
8. `recency_days`: Days elapsed since previous interaction (999.0 if no prior history).
9. `hist_category_preference`: 8-dimensional normalized histogram of previous category interactions.

*If $N_{hist} = 0$ (first interaction / cold start), all 9 behavioral features are set to 0.0 with an active cold-start indicator.*

---

## Simulated Interaction Layer Notice
> [!IMPORTANT]
> **Research Notice**:
> *"Simulated interaction data generated for research experimentation."*
> To evaluate sequence-aware behavioral transitions in offline experiments, simulated session steps `[View -> Add to Cart -> Purchase/Abandon]` are generated strictly from recorded funnel flags. No simulated sequences are claimed as collected from real-time production telemetry.
