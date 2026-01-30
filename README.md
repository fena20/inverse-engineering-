# سیستم پشتیبانی تصمیم برای بازسازی ساختمان (Retrofit DSS)

یک سیستم مدل‌سازی جایگزین (Surrogate) با ویژگی‌های «راهنما-یافته از فیزیک» برای پیش‌بینی عملکرد انرژی ساختمان و بهینه‌سازی رتروفیت با استفاده از داده‌های EPC چهار شهر انگلستان.

## Building Retrofit Decision Support System

A physics-guided (feature-engineered) surrogate modeling system for building energy performance prediction and retrofit optimization using UK EPC (Energy Performance Certificate) data from four major English cities.

---

## 📋 فهرست مطالب | Table of Contents

- [مقدمه](#مقدمه--overview)
- [ویژگی‌ها](#ویژگیها--features)
- [نصب](#نصب--installation)
- [استفاده](#استفاده--usage)
- [خروجی‌های پایان‌نامه](#خروجیهای-پایاننامه--thesis-outputs)
- [API](#api-endpoints)
- [عملکرد مدل](#عملکرد-مدل--model-performance)
- [ساختار پروژه](#ساختار-پروژه--project-structure)

---

## مقدمه | Overview

### مسئله
دشواری در تخمین سریع اثر تغییرات فیزیکی (پوسته و سیستم) بر مصرف انرژی و کربن بدون نیاز به شبیه‌سازی‌های سنگین دینامیک.

### هدف
ارائه یک موتور بهینه‌سازی که با دریافت هدف انرژی/کربن، ارزان‌ترین و اجرایی‌ترین مشخصات فنی (پوسته، تاسیسات، تجدیدپذیر) را پیشنهاد دهد.

### Problem Statement
Difficulty in quickly estimating the effect of physical changes (envelope and systems) on energy consumption and carbon without heavy dynamic simulations.

### Objective
Provide an optimization engine that, given energy/carbon targets, suggests the cheapest and most feasible technical specifications (envelope, HVAC, renewables).

---

## ویژگی‌ها | Features

### ✅ الزامات کارکردی پیاده‌سازی شده | Implemented Functional Requirements

| کد | الزام | وضعیت |
|----|-------|-------|
| FR-1 | ادغام داده‌های ۴ شهر با اثرات اقلیمی (HDD) | ✅ |
| FR-2 | تفسیرپذیری فیزیکی (اهمیت ویژگی‌های پوسته) | ✅ |
| FR-3 | تفکیک بارها (گرمایش، آبگرم، روشنایی) | ✅ |
| FR-4 | موتور پیشنهاد بر اساس INDICATIVE_COST | ✅ |

### 📊 داده‌های پشتیبانی شده | Supported Data

| شهر | تعداد رکورد | HDD | میانگین انرژی |
|-----|-------------|-----|---------------|
| Cambridge | 66,369 | 2,100 | 229 kWh/m² |
| Boston | 36,812 | 2,250 | 264 kWh/m² |
| Liverpool | 282,463 | 2,150 | 254 kWh/m² |
| Sheffield | 261,012 | 2,300 | 265 kWh/m² |
| **مجموع** | **646,656** | - | - |

---

## نصب | Installation

```bash
# کلون کردن مخزن
git clone https://github.com/fena20/inverse-engineering-.git
cd inverse-engineering-

# نصب وابستگی‌ها
pip install -r requirements.txt
```

---

## استفاده | Usage

### ۱. آموزش مدل‌ها | Train Models

```bash
python src/train.py --data-dir data --model-dir models --sample-size 100000
```

### ۲. تولید خروجی‌های پایان‌نامه | Generate Thesis Outputs

```bash
python src/analysis/thesis_analysis_fast.py
```

### ۳. اجرای مثال | Run Example

```bash
python src/example_usage.py
```

### ۴. راه‌اندازی API | Start API Server

```bash
cd src
python -m retrofit_dss.api.app
```

---

## خروجی‌های پایان‌نامه | Thesis Outputs

تمام نمودارها و جداول مورد نیاز برای پایان‌نامه در پوشه `outputs/thesis_figures/` تولید می‌شوند.

### فصل ۳: تحلیل داده‌ها (EDA)

| فایل | توضیح |
|------|-------|
| `fig3_1_city_energy_distribution.png` | توزیع مصرف انرژی در ۴ شهر (Box Plot) |
| `fig3_2_age_efficiency_heatmap.png` | نمودار حرارتی سن ساختمان vs بازدهی پوسته |
| `fig3_3_correlation_matrix.png` | ماتریس همبستگی ویژگی‌ها با خروجی‌ها |
| `table3_1_city_summary.csv` | خلاصه آماری داده‌های هر شهر |

### فصل ۵: نتایج مدل

| فایل | توضیح |
|------|-------|
| `fig5_1_actual_vs_predicted.png` | نمودار پراکندگی واقعی vs پیش‌بینی (به تفکیک شهر) |
| `fig5_2_residual_analysis.png` | تحلیل خطا به تفکیک شهر و سن ساختمان |
| `table5_1_model_accuracy.csv` | دقت مدل (R², MAE, RMSE) برای هر شهر |

### فصل ۶: تفسیرپذیری و اتصال به فیزیک

| فایل | توضیح |
|------|-------|
| `fig6_1_feature_importance.png` | اهمیت ویژگی‌ها برای ۴ مدل (انرژی، کربن، هزینه) |
| `fig6_2_sensitivity_analysis.png` | تحلیل حساسیت (دیوار، سقف، سیستم گرمایش) |

### فصل ۷: بهینه‌سازی و مهندسی معکوس

| فایل | توضیح |
|------|-------|
| `fig7_1_case_studies.png` | ۴ مطالعه موردی (قبل و بعد رتروفیت) |
| `fig7_2_pareto_curve.png` | منحنی هزینه-فایده پارتو |
| `fig7_3_recommended_measures.png` | ۱۵ اقدام پیشنهادی برتر از recommendations.csv |
| `table7_1_case_studies.csv` | جدول نتایج مطالعات موردی با هزینه‌ها |

---

## API Endpoints

### POST /evaluate
ارزیابی عملکرد ساختمان بر اساس مشخصات.

**درخواست:**
```json
{
  "building_profile": {
    "TOTAL_FLOOR_AREA": 90.0,
    "WALLS_ENERGY_EFF": "Poor",
    "ROOF_ENERGY_EFF": "Average",
    "MAINHEAT_ENERGY_EFF": "Good",
    "CONSTRUCTION_AGE_BAND": "England and Wales: 1930-1949",
    "PROPERTY_TYPE": "House",
    "BUILT_FORM": "Semi-Detached",
    "CITY": "Liverpool"
  }
}
```

**پاسخ:**
```json
{
  "energy_intensity_kwh_m2": 297.4,
  "carbon_intensity_kg_m2": 53.3,
  "heating_cost": 871,
  "hot_water_cost": 110,
  "lighting_cost": 101,
  "total_annual_cost": 1072,
  "epc_grade_estimate": "E"
}
```

### POST /optimize
دریافت توصیه‌های بهینه رتروفیت برای رسیدن به هدف.

**درخواست:**
```json
{
  "building_profile": { ... },
  "target_type": "carbon",
  "target_reduction": 50.0,
  "max_budget": 25000,
  "max_measures": 4
}
```

**پاسخ:**
```json
{
  "recommendations": [
    {
      "package_rank": 1,
      "measures": ["Loft insulation", "Cavity wall insulation", "Solar PV"],
      "total_cost_range": "£5,000 - £8,000",
      "energy_reduction_pct": 39.1,
      "carbon_reduction_pct": 50.2,
      "payback_years": 18.5
    }
  ]
}
```

### POST /sensitivity
تحلیل حساسیت برای یک پارامتر.

---

## عملکرد مدل | Model Performance

### دقت کلی | Overall Accuracy

| مدل | R² | MAE | RMSE | توضیح |
|-----|-----|-----|------|-------|
| Energy | 0.76 | 34.1 kWh/m² | 56.8 | شدت مصرف انرژی اولیه |
| Carbon | 0.56 | 5.9 kg/m² | 13.4 | شدت انتشار کربن |
| Heating Cost | 0.73 | £147 | £295 | هزینه سالانه گرمایش |
| Total Cost | 0.71 | £178 | £339 | کل هزینه سالانه انرژی |

### دقت به تفکیک شهر (مدل انرژی) | Per-City Accuracy

| شهر | R² | MAE | تعداد نمونه |
|-----|-----|-----|-------------|
| Cambridge | 0.86 | 32.3 | 421 |
| Boston | 0.86 | 36.3 | 628 |
| Liverpool | 0.68 | 36.8 | 4,291 |
| Sheffield | 0.81 | 31.2 | 4,073 |

### اعتبارسنجی فیزیکی | Physical Validation

✅ **سازگاری با شهود مهندسی (Sanity Check):**
- ویژگی‌های پوسته (دیوار، سقف) در رتبه‌های بالای اهمیت
- بهبود عایق‌کاری دیوار → کاهش ۲۳٪ مصرف انرژی (در تحلیل حساسیت)
- جلوگیری از نشت داده با تقسیم بر اساس Postcode و فیت‌کردن پیش‌پردازش روی Train

---

## ساختار پروژه | Project Structure

```
├── data/                          # داده‌های EPC شهرها
│   ├── domestic-E07000008-Cambridge/
│   ├── domestic-E07000136-Boston/
│   ├── domestic-E08000012-Liverpool/
│   └── domestic-E08000019-Sheffield/
│
├── src/
│   ├── retrofit_dss/              # پکیج اصلی
│   │   ├── data/                  # بارگذاری و پیش‌پردازش
│   │   │   ├── loader.py
│   │   │   └── preprocessor.py
│   │   ├── models/                # مدل‌های جایگزین
│   │   │   └── surrogate.py
│   │   ├── optimization/          # موتور بهینه‌سازی
│   │   │   └── engine.py
│   │   ├── api/                   # Flask REST API
│   │   │   └── app.py
│   │   └── utils/                 # ثوابت و توابع کمکی
│   │       ├── constants.py
│   │       └── helpers.py
│   │
│   ├── analysis/                  # اسکریپت‌های تحلیل پایان‌نامه
│   │   ├── thesis_analysis.py
│   │   └── thesis_analysis_fast.py
│   │
│   ├── train.py                   # آموزش مدل‌ها
│   └── example_usage.py           # مثال‌های کاربردی
│
├── outputs/
│   └── thesis_figures/            # نمودارها و جداول پایان‌نامه
│
├── models/                        # مدل‌های آموزش‌دیده
│   └── model_metrics.csv
│
├── requirements.txt
└── README.md
```

---

## موارد استفاده | Use Cases

### UC-1: پیش‌بینی عملکرد (Performance Prediction)
تخمین مصرف و هزینه بر اساس ویژگی‌های فعلی ساختمان.

### UC-2: طراحی معکوس (Inverse Design)
کاربر مقدار هدف (مثلاً ۶۰٪ کاهش کربن) را می‌دهد و سیستم ترکیب بهینه متغیرها را برمی‌گرداند.

### UC-3: تحلیل حساسیت (Sensitivity Analysis)
بررسی اثر تغییر یک پارامتر فیزیکی (مثل U-value دیوار) بر گرید نهایی EPC.

---

## محدودیت‌ها و ریسک‌ها | Limitations & Risks

1. **داده‌های ناقص**: بسیاری از رکوردهای EPC مشخصات دقیق پوسته را ندارند
2. **دقت هزینه**: هزینه‌های INDICATIVE_COST تقریبی هستند
3. **پوشش منطقه‌ای**: مدل روی ۴ شهر آموزش دیده و ممکن است به کل انگلستان تعمیم نیابد
4. **داده‌های سالانه**: فقط مصرف سالانه (نه ساعتی)

---

## منابع | References

- UK EPC Open Data: https://epc.opendatacommunitites.org/
- Open-Meteo Weather API: https://open-meteo.com/
- SAP Methodology: Standard Assessment Procedure for UK dwellings

---

## مجوز | License

مجوز استفاده از داده‌های EPC در فایل‌های LICENCE.txt در پوشه‌های data موجود است.

---

## تماس | Contact

برای سوالات و پیشنهادات، لطفاً Issue بزنید.
