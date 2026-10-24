FIELDS = (
    "member_id", "provider_name", "service_date", "procedure_code",
    "diagnosis_code", "requested_service", "decision_status",
)
PROVIDERS = ["Dr. Avery Stone", "Dr. Jordan Vale", "Dr. Morgan Reed", "Dr. Casey Park"]
SERVICES = ["Physical therapy", "MRI lumbar spine", "Office consultation", "Knee radiograph"]
STATUSES = ["Pending", "Approved", "Denied", "Under review"]
SEED = 20261024
LAYOUTS = ("intake", "authorization", "referral", "review")
HELD_OUT = "review"
