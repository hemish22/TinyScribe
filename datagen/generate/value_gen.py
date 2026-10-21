# Vendored from IndicFormBench (github.com/hemish22/IndicFormBench, src/generate/value_gen.py) at commit 6db4ae7f.
# TinyScribe changes are marked "TinyScribe:".
"""Realistic value generator (Phase 3.1) — Indian names, addresses, dates,
IDs, amounts, phone numbers, emails — in Latin, Devanagari, Bengali and Tamil.

Pure python, no deps. Deterministic per (seed): pass a random.Random instance.
"""
import re
import random
import unicodedata


DEVANAGARI = "अ आ इ ई उ ऊ ए ऐ ओ औ क ख ग घ च छ ज झ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह"

FIRST_M = ["Rohan", "Amit", "Vikram", "Suresh", "Rajesh", "Arjun", "Manish", "Deepak", "Sanjay", "Nikhil"]
FIRST_F = ["Priya", "Anjali", "Sunita", "Kavita", "Rekha", "Meena", "Pooja", "Neha", "Sarita", "Divya"]
FIRST_M_HI = ["रोहन", "अमित", "विक्रम", "सुरेश", "राजेश", "अर्जुन", "मनीष", "दीपक", "संजय", "निखिल"]
FIRST_F_HI = ["प्रिया", "अंजली", "सुनीता", "कविता", "रेखा", "मीना", "पूजा", "नेहा", "सरिता", "दिव्या"]
LAST = ["Sharma", "Verma", "Patel", "Gupta", "Kumar", "Singh", "Mishra", "Reddy", "Nair", "Yadav", "Mehta", "Joshi"]
LAST_HI = ["शर्मा", "वर्मा", "पटेल", "गुप्ता", "कुमार", "सिंह", "मिश्रा", "रेड्डी", "नायर", "यादव", "मेहता", "जोशी"]
CITY = ["Bhopal", "Jaipur", "Lucknow", "Patna", "Nagpur", "Indore", "Kanpur", "Surat", "Varanasi", "Udaipur"]
CITY_HI = ["भोपाल", "जयपुर", "लखनऊ", "पटना", "नागपुर", "इंदौर", "कानपुर", "सूरत", "वाराणसी", "उदयपुर"]
STATE = ["MP", "RJ", "UP", "BR", "MH", "GJ", "DL", "TN", "KA", "WB"]
STREET = ["Station Road", "Gandhi Marg", "New Colony", "Main Bazar", "Mirza Ismail Road", "Ring Road", "Civil Lines", "Sadar Bazar", "Linking Road", "Azad Nagar"]

N_PREFIXES = ["91", "98", "99", "70", "81", "88", "96"]
EMAIL_HOSTS = ["gmail.com", "yahoo.in", "outlook.com", "rediffmail.com", "hotmail.com"]


def name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_M + FIRST_F)} {rng.choice(LAST)}"


def name_hi(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_M_HI + FIRST_F_HI)} {rng.choice(LAST_HI)}"


def address(rng: random.Random) -> str:
    return f"{rng.randint(12, 489)} {rng.choice(STREET)}, {rng.choice(CITY)} - {rng.randint(10000, 799999)}"


def address_hi(rng: random.Random) -> str:
    return f"{rng.randint(12, 489)} {rng.choice(STREET)}, {rng.choice(CITY_HI)} - {rng.randint(10000, 799999)}"


def date(rng: random.Random) -> str:
    d = rng.randint(1, 28)
    m = rng.randint(1, 12)
    y = rng.randint(1965, 2008)
    return f"{d:02d}/{m:02d}/{y}"


def test_id(rng: random.Random) -> str:
    """TinyScribe: every identity number (Aadhaar, PAN, EPIC, TIN) is an obviously fake
    TEST-dddd, so the data holds no realistic-looking ID numbers."""
    return f"TEST-{rng.randint(0, 9999):04d}"


def id_number(rng: random.Random, kind: str = "aadhaar") -> str:
    return test_id(rng)  # TinyScribe: original generators below are unreachable
    if kind == "aadhaar":
        return f"{rng.randint(2, 9)}" + "".join(str(rng.randint(0, 9)) for _ in range(11))  # 12 digits, as on a real Aadhaar
    if kind == "pan":
        return rng.choice("ABCDEFGH") + rng.choice("PFE") + rng.choice("ABCDEFGHJKLMNP") + \
            rng.choice("ABCDEFGHJKLMNP") + rng.choice("ABCDEFGHJKLMNP") + f"{rng.randint(0, 9999):04d}"
    return str(rng.randint(2, 24)) + rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") * 2 + str(rng.randint(1, 999))


STATE_FULL = ["Madhya Pradesh", "Rajasthan", "Uttar Pradesh", "Bihar", "Maharashtra", "Gujarat", "Delhi", "Tamil Nadu", "Karnataka", "West Bengal"]
STATE_FULL_HI = ["मध्य प्रदेश", "राजस्थान", "उत्तर प्रदेश", "बिहार", "महाराष्ट्र", "गुजरात", "दिल्ली", "तमिल नाडु", "कर्नाटक", "पश्चिम बंगाल"]
STREET_HI = ["स्टेशन रोड", "गांधी मार्ग", "न्यू कॉलोनी", "मेन बाज़ार", "रिंग रोड", "सिविल लाइंस", "सदर बाज़ार"]
RELATION = ["Father", "Mother", "Brother", "Sister", "Spouse", "Son", "Daughter"]
RELATION_HI = ["पिता", "माता", "भाई", "बहन", "पति/पत्नी", "पुत्र", "पुत्री"]


def city(rng: random.Random) -> str:
    return rng.choice(CITY)


def city_hi(rng: random.Random) -> str:
    return rng.choice(CITY_HI)


def state(rng: random.Random) -> str:
    return rng.choice(STATE_FULL)


def state_hi(rng: random.Random) -> str:
    return rng.choice(STATE_FULL_HI)


def pincode(rng: random.Random) -> str:
    return str(rng.randint(110000, 799999))


def percent(rng: random.Random) -> str:
    return str(rng.randint(40, 90))


def epic(rng: random.Random) -> str:
    return test_id(rng)  # TinyScribe
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)) + f"{rng.randint(0, 9999999):07d}"


def serial(rng: random.Random) -> str:
    return str(rng.randint(1, 299))


def relation(rng: random.Random) -> str:
    return rng.choice(RELATION)


def relation_hi(rng: random.Random) -> str:
    return rng.choice(RELATION_HI)


def house(rng: random.Random) -> str:
    return f"{rng.randint(1, 489)}" + rng.choice(["", "/A", "/B", "-C"])


def street(rng: random.Random) -> str:
    return rng.choice(STREET)


def street_hi(rng: random.Random) -> str:
    return rng.choice(STREET_HI)


DOCUMENT = ["Ration Card", "School Leaving Certificate", "Voter ID of parent", "Electricity Bill", "Domicile Certificate", "Property Tax Receipt", "Gas Connection Bill"]
DOCUMENT_HI = ["राशन कार्ड", "स्कूल छोड़ने का प्रमाणपत्र", "माता-पिता का मतदाता पत्र", "बिजली का बिल", "अधिवास प्रमाणपत्र", "संपत्ति कर रसीद", "गैस कनेक्शन बिल"]
DISABILITY = ["Hearing impairment", "Low vision", "Cerebral palsy", "Autism spectrum", "Speech disability", "Dwarfism"]
DISABILITY_HI = ["श्रवण बाधित", "कम दृष्टि", "प्रमस्तिष्क घात", "स्वलीनता", "वाणी विकलांगता", "बौनापन"]


def document(rng: random.Random) -> str:
    return rng.choice(DOCUMENT)


def document_hi(rng: random.Random) -> str:
    return rng.choice(DOCUMENT_HI)


def disability(rng: random.Random) -> str:
    return rng.choice(DISABILITY)


def disability_hi(rng: random.Random) -> str:
    return rng.choice(DISABILITY_HI)


def amount(rng: random.Random) -> str:
    return f"{rng.randint(10, 9_99_999):,}"


def phone(rng: random.Random) -> str:
    return rng.choice(N_PREFIXES) + "".join(str(rng.randint(0, 9)) for _ in range(8))


def email(rng: random.Random) -> str:
    n = rng.choice(FIRST_M + FIRST_F + LAST).lower()
    return f"{n}{rng.randint(1, 99)}@{rng.choice(EMAIL_HOSTS)}"


def text(rng: random.Random, max_words: int = 40) -> str:
    return " ".join(rng.choice(
        ["humble", "necessary", "certified", "sought", "pertaining", "mentioned", "herewith", "enclosed",
         "applicant", "documents", "information", "particulars", "response", "grounds", "foregoing", "aforesaid"]
    ) for _ in range(rng.randint(3, max_words))) + "."


# value_type -> (latin_fn, devanagari_fn). Devanagari falls back to a transliterated
# name/address; structured strings (date/phone/amount/id) stay identical across scripts.
GENERATORS: dict = {
    "name":    (name, name_hi),
    "address": (address, address_hi),
    "date":    (date, lambda r: date(r)),
    "id_number": (lambda r: id_number(r), lambda r: id_number(r)),
    "amount":  (amount, amount),
    "phone":   (phone, phone),
    "email":   (email, lambda r: email(r)),
    "city":    (city, city_hi),
    "state":   (state, state_hi),
    "pincode": (pincode, pincode),
    "percent": (percent, percent),
    "epic":    (epic, epic),
    "serial":  (serial, serial),
    "relation": (relation, relation_hi),
    "house":   (house, house),
    "street":  (street, street_hi),
    "document": (document, document_hi),
    "disability": (disability, disability_hi),
    "text":    (text, lambda r: text(r)),
    "other":   (text, lambda r: text(r)),
}


# Bengali and Tamil (scope v3, Track B). Same value types as Hindi; structured strings
# (dates, phone, ids, amounts) stay in Latin digits, as on the printed forms. Tamil names
# follow the common given-name + father's-name pattern instead of a surname.
FIRST_M_BN = ["অমিত", "সৌরভ", "রাহুল", "অরিন্দম", "সুমন", "দেবাশিস", "প্রসেনজিৎ", "অভিজিৎ", "সুব্রত", "তন্ময়",
              "শুভজিৎ", "রাজীব", "সঞ্জয়", "অনির্বাণ", "পার্থ", "সুদীপ", "বিশ্বজিৎ", "কৌশিক", "সৌমিত্র", "অর্ণব"]
FIRST_F_BN = ["মৌমিতা", "সুস্মিতা", "প্রিয়াঙ্কা", "অনন্যা", "শ্রেয়া", "রুমা", "তনুশ্রী", "দেবযানী", "পাপিয়া", "সোমা",
              "রিয়া", "মধুমিতা", "ইন্দ্রাণী", "অর্পিতা", "সঙ্গীতা"]
LAST_BN = ["চট্টোপাধ্যায়", "মুখোপাধ্যায়", "বন্দ্যোপাধ্যায়", "ভট্টাচার্য", "দাস", "ঘোষ", "বসু", "সেন", "রায়", "মণ্ডল",
           "সাহা", "পাল", "দত্ত", "সরকার", "বিশ্বাস", "চক্রবর্তী", "মজুমদার", "হালদার", "নস্কর", "প্রামাণিক"]
CITY_BN = ["কলকাতা", "হাওড়া", "দুর্গাপুর", "আসানসোল", "শিলিগুড়ি", "বর্ধমান", "মালদা", "বহরমপুর", "কৃষ্ণনগর", "খড়্গপুর",
           "মেদিনীপুর", "বাঁকুড়া", "পুরুলিয়া", "জলপাইগুড়ি", "কোচবিহার", "বারাসাত", "হুগলি", "শ্রীরামপুর", "বোলপুর", "দার্জিলিং"]
STATE_BN = ["পশ্চিমবঙ্গ", "বিহার", "ওড়িশা", "অসম", "ত্রিপুরা", "ঝাড়খণ্ড", "দিল্লি", "মহারাষ্ট্র"]
STREET_BN = ["স্টেশন রোড", "নেতাজি সুভাষ রোড", "রবীন্দ্র সরণি", "গান্ধী রোড", "বিবেকানন্দ পল্লী", "নেহরু কলোনি",
             "বাজার পাড়া", "কলেজ রোড"]
RELATION_BN = ["পিতা", "মাতা", "স্বামী", "স্ত্রী", "পুত্র", "কন্যা", "ভাই", "বোন"]
DOCUMENT_BN = ["আধার কার্ড", "রেশন কার্ড", "প্যান কার্ড", "ভোটার কার্ড", "বিদ্যুৎ বিল", "পাসপোর্ট", "ব্যাঙ্কের পাসবই"]
DISABILITY_BN = ["দৃষ্টিহীন", "শ্রবণ প্রতিবন্ধী", "বাক প্রতিবন্ধী", "চলন প্রতিবন্ধী"]

FIRST_M_TA = ["முருகன்", "கார்த்திக்", "செந்தில்", "அருண்", "விஜய்", "சரவணன்", "ராஜேஷ்", "பாலாஜி", "சுரேஷ்", "கணேஷ்",
              "மணிகண்டன்", "பிரகாஷ்", "வெங்கடேஷ்", "அசோக்", "தினேஷ்", "ரமேஷ்", "சிவக்குமார்", "அருள்", "கண்ணன்", "பாண்டியன்"]
FIRST_F_TA = ["லட்சுமி", "மீனா", "கவிதா", "பிரியா", "சரண்யா", "தேவி", "ஜெயந்தி", "வசந்தி", "அனிதா", "கலைச்செல்வி",
              "சித்ரா", "பூங்கொடி", "செல்வி", "ராதா", "மகேஸ்வரி"]
LAST_TA = ["செல்வராஜ்", "சுப்பிரமணியம்", "ராமசாமி", "கிருஷ்ணன்", "பழனிசாமி", "வேலு", "நடராஜன்", "சண்முகம்", "அண்ணாமலை",
           "பெருமாள்", "ஆறுமுகம்", "தங்கராஜ்", "கந்தசாமி", "சேகர்", "ராஜேந்திரன்"]
CITY_TA = ["சென்னை", "மதுரை", "கோயம்புத்தூர்", "திருச்சிராப்பள்ளி", "சேலம்", "திருநெல்வேலி", "வேலூர்", "ஈரோடு",
           "தூத்துக்குடி", "தஞ்சாவூர்", "திண்டுக்கல்", "காஞ்சிபுரம்", "கரூர்", "நாகர்கோவில்", "கடலூர்", "விழுப்புரம்",
           "திருப்பூர்", "ஓசூர்", "கும்பகோணம்", "புதுக்கோட்டை"]
STATE_TA = ["தமிழ்நாடு", "கேரளா", "கர்நாடகா", "ஆந்திரப் பிரதேசம்", "புதுச்சேரி", "தெலங்கானா", "மகாராஷ்டிரா", "டெல்லி"]
STREET_TA = ["காந்தி சாலை", "நேரு தெரு", "பெரியார் நகர்", "அண்ணா சாலை", "கோயில் தெரு", "மெயின் ரோடு", "பஜார் தெரு",
             "காமராஜர் சாலை"]
RELATION_TA = ["தந்தை", "தாய்", "கணவர்", "மனைவி", "மகன்", "மகள்", "சகோதரர்", "சகோதரி"]
DOCUMENT_TA = ["ஆதார் அட்டை", "குடும்ப அட்டை", "பான் அட்டை", "வாக்காளர் அடையாள அட்டை", "மின் கட்டண ரசீது",
               "கடவுச்சீட்டு", "வங்கி கணக்கு புத்தகம்"]
DISABILITY_TA = ["பார்வைக் குறைபாடு", "செவித்திறன் குறைபாடு", "பேச்சுக் குறைபாடு", "கை கால் ஊனம்"]


# "other" = a government scheme name (the blanks on scheme application forms), "text" = a short
# free-text answer (qualification, remark), both in the form's own language
OTHER_BN = ["রূপশ্রী", "কন্যাশ্রী", "ঐক্যশ্রী", "স্বাস্থ্য সাথী", "লক্ষ্মীর ভাণ্ডার", "কৃষক বন্ধু", "সবুজ সাথী",
            "জয় বাংলা", "তরুণের স্বপ্ন", "খাদ্য সাথী"]
TEXT_BN = ["মাধ্যমিক পাশ", "উচ্চ মাধ্যমিক পাশ", "স্নাতক", "স্নাতকোত্তর", "প্রয়োজনীয় নথি সংযুক্ত করা হল",
           "আবেদনপত্র জমা দেওয়া হয়েছে", "কোনও তথ্য নেই", "পূর্বে আবেদন করা হয়নি"]
OTHER_TA = ["புதுமைப் பெண் திட்டம்", "மகளிர் உரிமைத் தொகை", "முதலமைச்சரின் விரிவான மருத்துவக் காப்பீட்டுத் திட்டம்",
            "முதியோர் உதவித்தொகை", "இலவச வீட்டுமனைப் பட்டா", "திருமண உதவித் திட்டம்", "கல்வி உதவித்தொகை"]
TEXT_TA = ["பத்தாம் வகுப்பு தேர்ச்சி", "பன்னிரண்டாம் வகுப்பு தேர்ச்சி", "இளங்கலை பட்டம்", "முதுகலை பட்டம்",
           "சான்றிதழ் இணைக்கப்பட்டுள்ளது", "விவரம் இல்லை", "முன்பு விண்ணப்பிக்கவில்லை"]


def _script_gens(first_m, first_f, last, city_, state_, street_, relation_, document_, disability_,
                 other_=(), text_=()) -> dict:
    return {
        "name": lambda r: f"{r.choice(first_m + first_f)} {r.choice(last)}",
        "address": lambda r: f"{r.randint(12, 489)} {r.choice(street_)}, {r.choice(city_)} - {r.randint(110000, 799999)}",
        "city": lambda r: r.choice(city_),
        "state": lambda r: r.choice(state_),
        "street": lambda r: r.choice(street_),
        "relation": lambda r: r.choice(relation_),
        "document": lambda r: r.choice(document_),
        "disability": lambda r: r.choice(disability_),
        "other": lambda r: r.choice(other_),
        "text": lambda r: r.choice(text_),
    }


# value_type -> generator for the native-script types; every other type uses the Latin generator
SCRIPT_GENERATORS: dict = {
    "bengali": _script_gens(FIRST_M_BN, FIRST_F_BN, LAST_BN, CITY_BN, STATE_BN, STREET_BN, RELATION_BN,
                            DOCUMENT_BN, DISABILITY_BN, OTHER_BN, TEXT_BN),
    "tamil": _script_gens(FIRST_M_TA, FIRST_F_TA, LAST_TA, CITY_TA, STATE_TA, STREET_TA, RELATION_TA,
                          DOCUMENT_TA, DISABILITY_TA, OTHER_TA, TEXT_TA),
}


# Handwriting mode (handwriting.py): every Devanagari word must have real handwritten samples in
# IIIT-HW-Dev (tools/hw_datasets/iiit_hw_hindi/vocab.tsv), so these lists are drawn from that
# vocabulary instead of the lists above. Latin parts and digits come from EMNIST glyphs.
FIRST_M_HW = ["राम", "राहुल", "राजू", "अजय", "विजय", "संजय", "सुनील", "राजेश", "महेश", "गणेश", "दिनेश",
              "रमेश", "राकेश", "आकाश", "प्रमोद", "मनोज", "सुभाष", "किशन", "शंकर", "भरत", "अभिषेक",
              "अमन", "अंकित", "आशीष", "हर्ष", "प्रवीण", "रजत", "योगेश", "जगदीश", "रितेश"]
FIRST_F_HW = ["सीता", "गीता", "राधा", "सुधा", "उषा", "आशा", "निशा", "पूजा", "कमला", "ज्योति", "सुमन",
              "रेणु", "सुनीता", "संगीता", "सरिता", "लक्ष्मी", "सरस्वती", "गंगा", "यमुना", "मधु", "अर्चना", "कल्पना"]
LAST_HW = ["शर्मा", "मिश्रा", "तिवारी", "पांडे", "त्रिपाठी", "जैन", "अग्रवाल", "मित्तल", "चौहान", "ठाकुर",
           "प्रसाद", "चौधरी", "नेगी"]
CITY_HW = ["दिल्ली", "कोलकाता", "जयपुर", "कानपुर", "इंदौर", "नागपुर", "आगरा", "मेरठ", "इलाहाबाद", "गोरखपुर",
           "अजमेर", "उदयपुर", "जोधपुर", "बीकानेर", "कोटा", "रांची", "देहरादून", "अमृतसर", "चंडीगढ़", "अयोध्या"]
STATE_HW = ["राजस्थान", "बिहार", "दिल्ली", "हरियाणा", "झारखंड", "उत्तराखंड", "गोवा", "कर्नाटक"]
STREET_HW = ["सदर बाजार", "गांधी पथ", "नेहरू पथ", "सुभाष पथ"]
RELATION_HW = ["माता", "माँ", "पत्नी", "पुत्र", "पुत्री", "दादा", "नानी", "मामा"]
DOCUMENT_HW = ["राशन कार्ड", "आधार कार्ड", "पैन कार्ड"]
DISABILITY_HW = ["श्रवण बाधित", "दृष्टि बाधित", "वाणी बाधित", "अंधापन"]

HW_DEVANAGARI: dict = {
    "name":    lambda r: f"{r.choice(FIRST_M_HW + FIRST_F_HW)} {r.choice(LAST_HW)}",
    "address": lambda r: f"{r.randint(12, 489)} {r.choice(STREET)}, {r.choice(CITY_HW)} - {r.randint(10000, 799999)}",
    "city":    lambda r: r.choice(CITY_HW),
    "state":   lambda r: r.choice(STATE_HW),
    "relation": lambda r: r.choice(RELATION_HW),
    "street":  lambda r: r.choice(STREET_HW),
    "document": lambda r: r.choice(DOCUMENT_HW),
    "disability": lambda r: r.choice(DISABILITY_HW),
}


# Table cell types (scope v3, Track A): household, education, employment and FATCA tables.
# Numbers (age, year, marks) stay in Latin digits in every script, as on the printed forms.
GENDER = {"latin": ["Male", "Female"], "devanagari": ["पुरुष", "महिला"],
          "bengali": ["পুরুষ", "মহিলা"], "tamil": ["ஆண்", "பெண்"]}
OCCUPATION = {
    "latin": ["Farmer", "Student", "Labourer", "Housewife", "Shopkeeper", "Teacher", "Driver", "Tailor",
              "Clerk", "Retired"],
    "devanagari": ["किसान", "छात्र", "मज़दूर", "गृहिणी", "दुकानदार", "शिक्षक", "ड्राइवर", "दर्जी", "क्लर्क",
                   "सेवानिवृत्त"],
    "bengali": ["কৃষক", "ছাত্র", "শ্রমিক", "গৃহবধূ", "দোকানদার", "শিক্ষক", "চালক", "দর্জি", "কেরানি",
                "অবসরপ্রাপ্ত"],
    "tamil": ["விவசாயி", "மாணவர்", "கூலித் தொழிலாளி", "இல்லத்தரசி", "கடை உரிமையாளர்", "ஆசிரியர்",
              "ஓட்டுநர்", "தையல்காரர்", "எழுத்தர்", "ஓய்வு பெற்றவர்"],
}
BOARD = {
    "latin": ["CBSE", "ICSE", "MP Board", "UP Board", "RBSE", "Delhi University", "Anna University",
              "Calcutta University", "IGNOU", "Rajasthan University"],
    "devanagari": ["सीबीएसई", "माध्यमिक शिक्षा मंडल", "उत्तर प्रदेश बोर्ड", "दिल्ली विश्वविद्यालय",
                   "राजस्थान विश्वविद्यालय", "इग्नू", "पटना विश्वविद्यालय"],
    "bengali": ["পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ", "উচ্চ মাধ্যমিক শিক্ষা সংসদ", "কলকাতা বিশ্ববিদ্যালয়",
                "বর্ধমান বিশ্ববিদ্যালয়", "যাদবপুর বিশ্ববিদ্যালয়", "সিবিএসই"],
    "tamil": ["தமிழ்நாடு அரசு தேர்வுகள் இயக்ககம்", "சென்னைப் பல்கலைக்கழகம்", "அண்ணா பல்கலைக்கழகம்",
              "மதுரை காமராசர் பல்கலைக்கழகம்", "பாரதிதாசன் பல்கலைக்கழகம்", "சிபிஎஸ்இ"],
}
EMPLOYER = {
    "latin": ["Govt. Primary School", "District Hospital", "Gram Panchayat", "Post Office",
              "Municipal Corporation", "Cooperative Society", "Sharma Traders", "City Hospital"],
    "devanagari": ["राजकीय प्राथमिक विद्यालय", "ज़िला अस्पताल", "ग्राम पंचायत", "डाकघर", "नगर निगम", "सहकारी समिति"],
    "bengali": ["সরকারি প্রাথমিক বিদ্যালয়", "জেলা হাসপাতাল", "গ্রাম পঞ্চায়েত", "ডাকঘর", "পৌরসভা", "সমবায় সমিতি"],
    "tamil": ["அரசு தொடக்கப் பள்ளி", "மாவட்ட மருத்துவமனை", "கிராம ஊராட்சி", "அஞ்சல் அலுவலகம்", "நகராட்சி",
              "கூட்டுறவுச் சங்கம்"],
}
DESIGNATION = {
    "latin": ["Assistant", "Clerk", "Teacher", "Accountant", "Data Entry Operator", "Supervisor", "Nurse",
              "Peon"],
    "devanagari": ["सहायक", "लिपिक", "शिक्षक", "लेखाकार", "पर्यवेक्षक", "नर्स", "चपरासी"],
    "bengali": ["সহকারী", "কেরানি", "শিক্ষক", "হিসাবরক্ষক", "সুপারভাইজার", "নার্স"],
    "tamil": ["உதவியாளர்", "எழுத்தர்", "ஆசிரியர்", "கணக்காளர்", "மேற்பார்வையாளர்", "செவிலியர்"],
}
COUNTRY = {
    "latin": ["India", "Nepal", "United States", "United Kingdom", "UAE", "Singapore", "Canada", "Australia"],
    "devanagari": ["भारत", "नेपाल", "अमेरिका", "ब्रिटेन", "कनाडा", "सिंगापुर"],
    "bengali": ["ভারত", "নেপাল", "আমেরিকা", "ব্রিটেন", "কানাডা", "বাংলাদেশ"],
    "tamil": ["இந்தியா", "நேபாளம்", "அமெரிக்கா", "பிரிட்டன்", "கனடா", "சிங்கப்பூர்"],
}
YES_NO = {"latin": ["Yes", "No"], "devanagari": ["हाँ", "नहीं"], "bengali": ["হ্যাঁ", "না"],
          "tamil": ["ஆம்", "இல்லை"]}
MAX_MARKS = [100, 500, 600, 1000, 1200]


def tin(rng: random.Random) -> str:
    """A taxpayer id: US SSN-style, or a 10-character PAN-style id."""
    if rng.random() < 0.5:
        return f"{rng.randint(100, 899)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
    return ("".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5))
            + f"{rng.randint(0, 9999):04d}" + rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"))


def _table_gens(script: str) -> dict:
    pick = lambda d: (lambda r: r.choice(d[script]))
    return {
        "age": lambda r: str(r.randint(1, 85)),
        "year": lambda r: str(r.randint(1988, 2024)),
        "max_marks": lambda r: str(r.choice(MAX_MARKS)),
        "marks": lambda r: str(r.randint(150, 480)),
        "tin": test_id,  # TinyScribe: was tin
        "gender": pick(GENDER), "occupation": pick(OCCUPATION), "board": pick(BOARD),
        "employer": pick(EMPLOYER), "designation": pick(DESIGNATION), "country": pick(COUNTRY),
        "yes_no": pick(YES_NO),
    }


for _t, _fn in _table_gens("latin").items():
    GENERATORS[_t] = (_fn, _table_gens("devanagari")[_t])
for _s in ("bengali", "tamil"):
    SCRIPT_GENERATORS[_s].update(_table_gens(_s))

# handwriting: only words with IIIT samples. Gender and school boards are written the way
# people fill Hindi forms when the word is not at hand: M/F and Latin board abbreviations.
OCCUPATION_HW = ["किसान", "मज़दूर", "ड्राइवर", "सेवानिवृत्त", "व्यापारी", "डॉक्टर", "पुजारी", "सैनिक"]
BOARD_HW = ["CBSE", "RBSE", "दिल्ली विश्वविद्यालय", "राजस्थान विश्वविद्यालय", "आगरा विश्वविद्यालय",
            "कानपुर विश्वविद्यालय", "इलाहाबाद विश्वविद्यालय", "इग्नू"]
EMPLOYER_HW = ["राजकीय विद्यालय", "राजकीय महाविद्यालय", "डाकघर", "पंचायत समिति", "पंचायत कार्यालय"]
DESIGNATION_HW = ["लिपिक", "अधिकारी", "चालक"]
COUNTRY_HW = ["जापान", "चीन", "INDIA", "NEPAL", "USA"]
HW_DEVANAGARI.update({
    "gender": lambda r: r.choice(["M", "F"]),
    "occupation": lambda r: r.choice(OCCUPATION_HW),
    "board": lambda r: r.choice(BOARD_HW),
    "employer": lambda r: r.choice(EMPLOYER_HW),
    "designation": lambda r: r.choice(DESIGNATION_HW),
    "country": lambda r: r.choice(COUNTRY_HW),
    "yes_no": lambda r: r.choice(["हाँ", "NO"]),
})


# TinyScribe: Bengali and Tamil handwriting. Words with real samples (train >= 2, test >= 1) in the
# Tamil IIIT bank and in BanglaWriting (tools/hw_datasets/*/vocab.tsv). Coverage is thin (BanglaWriting
# is free-text essays, IIIT Tamil is news vocabulary), so a handwritten Bengali/Tamil field uses these
# lists HW_INDIC_P of the time; other values are drawn in the script's handwriting font (handwriting.py).
HW_INDIC_P = 0.6
FIRST_M_HW_TA = ["செந்தில்", "அருண்", "விஜய்", "அருள்", "சிவா", "சேகர்", "சுந்தர்", "விக்ரம்", "அர்ஜுன்"]
FIRST_F_HW_TA = ["பிரியா", "அனிதா", "செல்வி", "ராதா", "சீதா", "வள்ளி", "ரேவதி", "சுமதி", "பத்மா", "விஜயா",
                 "நிர்மலா", "திவ்யா"]
LAST_HW_TA = ["வேலு", "சேகர்", "நாடார்"] + FIRST_M_HW_TA   # Tamil names often carry the father's name
CITY_HW_TA = ["திருச்சி", "தஞ்சை", "கரூர்", "கடலூர்", "திருப்பூர்", "பழனி"]
RELATION_HW_TA = ["கணவர்", "மனைவி", "தாத்தா"]
COUNTRY_HW_TA = ["சிங்கப்பூர்", "பிரிட்டன்"]
RELATION_HW_BN = ["বাবা", "মা", "ভাই", "নানা", "মামা"]
OCCUPATION_HW_BN = ["দোকানদার", "শিক্ষক", "চাকরি", "ব্যবসা", "মজুর"]
COUNTRY_HW_BN = ["ভারত", "বাংলাদেশ", "চীন", "জাপান"]
HW_INDIC: dict = {
    "tamil": {
        "name": lambda r: f"{r.choice(FIRST_M_HW_TA + FIRST_F_HW_TA)} {r.choice(LAST_HW_TA)}",
        "city": lambda r: r.choice(CITY_HW_TA),
        "relation": lambda r: r.choice(RELATION_HW_TA),
        "country": lambda r: r.choice(COUNTRY_HW_TA),
    },
    "bengali": {
        "relation": lambda r: r.choice(RELATION_HW_BN),
        "occupation": lambda r: r.choice(OCCUPATION_HW_BN),
        "country": lambda r: r.choice(COUNTRY_HW_BN),
    },
}


def hw_indic_words(script: str) -> set:
    """Every native-script word a Bengali/Tamil value can contain: the HW_INDIC lists plus the
    regular lists (their words get real handwriting whenever the bank has them)."""
    lo, hi = {"bengali": ("\u0980", "\u09ff"), "tamil": ("\u0b80", "\u0bff")}[script]
    sfx = {"bengali": "_BN", "tamil": "_TA"}[script]
    lists = [v for k, v in globals().items() if isinstance(v, list) and k.endswith(sfx)]
    lists += [d[script] for d in (GENDER, OCCUPATION, BOARD, EMPLOYER, DESIGNATION, COUNTRY, YES_NO)]
    return {unicodedata.normalize("NFC", w) for lst in lists for x in lst for w in x.split()
            if any(lo <= c <= hi for c in w)}


def table_row_values(types: list, script: str, rng: random.Random, handwriting: bool = False,
                     prev: list | None = None) -> list:
    """Values for one table row (value_type per cell, None for printed cells), kept consistent
    within the row (marks <= maximum marks with the percentage computed from them, a from-date
    before a to-date) and with `prev`, the row above: a later year of passing, a period that
    starts after the previous one ends."""
    vals = [generate_value(t, script, rng, handwriting=handwriting) if t else None for t in types]
    if "year" in types:
        i = types.index("year")
        last = prev[i] if prev and prev[i] and prev[i].isdigit() else None
        vals[i] = str(min(2025, int(last) + rng.randint(2, 3))) if last else str(rng.randint(1990, 2012))
    if "max_marks" in types and "marks" in types:
        mx = rng.choice(MAX_MARKS)
        got = rng.randint(int(mx * 0.38), int(mx * 0.96))
        vals[types.index("max_marks")], vals[types.index("marks")] = str(mx), str(got)
        if "percent" in types:
            vals[types.index("percent")] = f"{100 * got / mx:.2f}"
    dates = [i for i, t in enumerate(types) if t == "date"]
    if len(dates) == 2:
        months = lambda d: int(d[-4:]) * 12 + int(d[3:5]) - 1
        fmt = lambda m: f"{rng.randint(1, 28):02d}/{m % 12 + 1:02d}/{m // 12}"
        last = prev[dates[1]] if prev and prev[dates[1]] else None
        start = months(last) + rng.randint(1, 10) if last else rng.randint(1998 * 12, 2014 * 12)
        start = min(start, 2024 * 12)
        end = min(start + rng.randint(10, 60), 2025 * 12 + 8)
        vals[dates[0]], vals[dates[1]] = fmt(start), fmt(end)
    return vals


def hw_hindi_words() -> set:
    lists = [FIRST_M_HW, FIRST_F_HW, LAST_HW, CITY_HW, STATE_HW, STREET_HW, RELATION_HW, DOCUMENT_HW, DISABILITY_HW,
             OCCUPATION_HW, BOARD_HW, EMPLOYER_HW, DESIGNATION_HW, COUNTRY_HW]
    return {w for lst in lists for x in lst for w in x.split() if any("ऀ" <= c <= "ॿ" for c in w)}


# one akshara: (consonant + nukta? + halant)* then a consonant/vowel, nukta, matra, and
# anusvara/chandrabindu/visarga (or a trailing halant). Bengali has the same block layout
# (+0x80), plus khanda ta and the au length mark. Tamil does not stack conjuncts: each
# letter (consonant + vowel sign or pulli) takes its own box, except க்ஷ and ஸ்ரீ.
AKSHARA = re.compile(
    r"(?:[\u0915-\u0939\u0958-\u095f]\u093c?\u094d)*[\u0904-\u0939\u0958-\u0961\u0972-\u097f]"
    r"\u093c?[\u093e-\u094c\u0962\u0963]?[\u0900-\u0903]?\u094d?"
    r"|(?:[\u0995-\u09b9\u09dc-\u09df]\u09bc?\u09cd)*[\u0985-\u09b9\u09ce\u09dc-\u09e1]"
    r"\u09bc?[\u09be-\u09cc\u09e2\u09e3]?\u09d7?[\u0981-\u0983]?\u09cd?"
    r"|\u0b95\u0bcd\u0bb7[\u0bbe-\u0bcd]?|\u0bb8\u0bcd\u0bb0\u0bc0"
    r"|[\u0b95-\u0bb9][\u0bbe-\u0bcd]?\u0bd7?|[\u0b82\u0b83\u0b85-\u0b94]"
    r"|.", re.S)


def aksharas(value: str) -> list:
    """Split into comb cells: one akshara (syllable cluster) per cell, as a person fills
    a Devanagari/Bengali/Tamil comb; any other character (digit, space, Latin) is its own cell."""
    return AKSHARA.findall(value)


def generate_value(value_type: str, label_script: str, rng: random.Random, handwriting: bool = False):
    """Return a ground-truth value for a field.
    label_script in {latin, devanagari, bilingual, bengali, tamil, none} (or a field's value_script).
    Devanagari/Bilingual → Devanagari value; bengali/tamil → that script, NFC-normalised."""
    latin_fn, dev_fn = GENERATORS.get(value_type, GENERATORS["text"])
    if label_script in SCRIPT_GENERATORS:
        fn = SCRIPT_GENERATORS[label_script].get(value_type, latin_fn)
        hw = HW_INDIC.get(label_script, {})
        if handwriting and value_type in hw and rng.random() < HW_INDIC_P:  # TinyScribe
            fn = hw[value_type]
        return unicodedata.normalize("NFC", fn(rng))
    if label_script in ("devanagari", "bilingual"):
        if handwriting and value_type in HW_DEVANAGARI:
            return HW_DEVANAGARI[value_type](rng)
        return dev_fn(rng)
    return latin_fn(rng)