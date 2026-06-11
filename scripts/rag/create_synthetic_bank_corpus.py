from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "data/rag_corpus"

DOCUMENTS = {
    "account_opening_en.txt": """Account Opening Policy

Customers can open a savings account by submitting a valid national ID or passport, proof of address, and a completed account opening form.

The minimum age for opening an individual savings account is 18 years. Customers under 18 may open a minor account with a legal guardian.

The bank may request additional documents if identity verification or compliance checks require it.
""",
    "account_opening_ar.txt": """سياسة فتح الحساب

يمكن للعميل فتح حساب توفير من خلال تقديم بطاقة رقم قومي أو جواز سفر ساري، وإثبات عنوان، ونموذج فتح حساب مكتمل.

الحد الأدنى لعمر فتح حساب توفير فردي هو 18 سنة. يمكن لمن هم أقل من 18 سنة فتح حساب قاصر بموافقة ولي الأمر.

قد يطلب البنك مستندات إضافية إذا كانت هناك حاجة للتحقق من الهوية أو إجراءات الامتثال.
""",
    "card_replacement_en.txt": """Debit Card Replacement Policy

If a debit card is lost, stolen, or damaged, the customer should contact the bank immediately to block the card.

A replacement card can be requested through the branch, mobile banking app, or customer service hotline.

Replacement fees may apply depending on the account package. The new card is usually available within 3 to 5 business days.
""",
    "card_replacement_ar.txt": """سياسة استبدال بطاقة الخصم

إذا فقد العميل بطاقة الخصم أو تعرضت للسرقة أو التلف، يجب عليه التواصل مع البنك فوراً لإيقاف البطاقة.

يمكن طلب بطاقة بديلة من خلال الفرع أو تطبيق الهاتف المحمول أو خدمة العملاء.

قد يتم تطبيق رسوم استبدال حسب نوع الحساب. عادةً تكون البطاقة الجديدة متاحة خلال 3 إلى 5 أيام عمل.
""",
    "loan_policy_en.txt": """Personal Loan Policy

Personal loan eligibility depends on income, employment status, credit history, and internal risk assessment.

Customers must provide proof of income, valid identification, and any additional documents requested by the bank.

Loan approval is not guaranteed. The final decision depends on the bank's credit policy and affordability checks.
""",
    "loan_policy_ar.txt": """سياسة القرض الشخصي

تعتمد أهلية الحصول على القرض الشخصي على الدخل، وحالة العمل، والتاريخ الائتماني، وتقييم المخاطر الداخلي.

يجب على العميل تقديم إثبات دخل، ومستند هوية ساري، وأي مستندات إضافية يطلبها البنك.

الموافقة على القرض ليست مضمونة، ويعتمد القرار النهائي على سياسة الائتمان وقدرة العميل على السداد.
""",
}

def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, content in DOCUMENTS.items():
        path = CORPUS_DIR / filename
        path.write_text(content.strip() + "\n", encoding="utf-8")
        print("Created:", path)

    print("\nSynthetic banking corpus created successfully.")
    print("Total documents:", len(DOCUMENTS))

if __name__ == "__main__":
    main()
