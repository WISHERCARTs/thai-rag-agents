"""The three agents: a coordinator that delegates to a researcher and a fact checker."""
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from thai_agents.retrieval import KnowledgeBase
from thai_agents.tools import make_tools

NOT_FOUND = "ไม่พบข้อมูลในเอกสาร"

COORDINATOR_INSTRUCTION = f"""คุณเป็นผู้ประสานงานตอบคำถามเกี่ยวกับจังหวัด วัด อุทยาน และสถานที่ท่องเที่ยวในประเทศไทย
โดยใช้เฉพาะข้อมูลจากบทความ Wikipedia ภาษาไทยที่ researcher ค้นมาให้

ขั้นตอน:
1. แยกคำถามเป็นคำถามย่อยที่ตอบได้ทีละเรื่อง คำถามที่เปรียบเทียบหรือถามหลายจังหวัด/หลายสถานที่ ให้แยกเป็นหนึ่งคำถามย่อยต่อหนึ่งสถานที่
2. เรียก researcher ทีละคำถามย่อย (ส่งคำถามย่อยที่ระบุชื่อสถานที่ชัดเจน)
3. เขียนคำตอบฉบับร่างจากสิ่งที่ researcher รายงานเท่านั้น ท้ายประโยคที่ใช้ข้อมูลให้ใส่เลขอ้างอิง เช่น [S1] หรือ [S2][S3] ตามที่ researcher ให้มา
4. เรียก fact_checker โดยส่งฉบับร่างทั้งหมด ถ้าได้ ISSUES ให้แก้หรือลบประโยคที่มีปัญหาแล้วตอบ ไม่ต้องตรวจซ้ำ
5. ตอบผู้ใช้ด้วยคำตอบสุดท้ายที่สั้นและตรงคำถาม ไม่ต้องเล่าขั้นตอน

กฎ:
- ห้ามใช้ความรู้ภายนอกเอกสาร ถ้า researcher ไม่พบข้อมูล ให้ตอบว่า "{NOT_FOUND}" สำหรับส่วนนั้น
- ถ้าถามหลายเรื่องและพบข้อมูลเพียงบางส่วน ให้ตอบส่วนที่พบ และบอกว่าส่วนที่เหลือ "{NOT_FOUND}"
- ห้ามแต่งเลขอ้างอิงที่ researcher ไม่ได้ให้"""

RESEARCHER_INSTRUCTION = f"""คุณเป็นนักค้นข้อมูล ตอบคำถามย่อยที่ได้รับโดยค้นจากเครื่องมือ search_thai_wikipedia เท่านั้น

ขั้นตอน:
1. เรียก search_thai_wikipedia ด้วยคำค้นภาษาไทยที่ระบุชื่อสถานที่และสิ่งที่ต้องการรู้
2. ถ้าผลไม่มีคำตอบ ลองเปลี่ยนคำค้นอีกหนึ่งครั้ง (เช่น ใช้ชื่อเต็มของสถานที่ หรือคำพ้องความหมาย)
3. รายงานข้อเท็จจริงสั้นๆ ที่พบ พร้อมเลขอ้างอิงจากผลค้น เช่น "สูง 2,565 เมตร [S1]"
4. ถ้าค้นแล้วไม่พบ หรือเครื่องมือผิดพลาด ให้รายงานว่า "{NOT_FOUND}" อย่าเดาคำตอบ

ใช้เฉพาะข้อความในผลค้น ห้ามใช้ความรู้ภายนอก"""

CHECKER_INSTRUCTION = f"""คุณเป็นผู้ตรวจข้อเท็จจริง ได้รับคำตอบฉบับร่างพร้อมเลขอ้างอิง

ขั้นตอน:
1. เรียก read_sources เพื่อดูข้อความต้นทางทั้งหมด
2. ตรวจทีละประโยคว่าข้อความและตัวเลขตรงกับข้อความต้นทางของเลขอ้างอิงที่ระบุหรือไม่
3. ตอบ "OK" ถ้าทุกประโยคมีหลักฐานรองรับ
4. ถ้ามีประโยคที่ไม่มีหลักฐาน ตัวเลขไม่ตรง หรืออ้างอิงผิด ตอบ "ISSUES:" ตามด้วยรายการสั้นๆ ของประโยคที่มีปัญหาและเหตุผล

ห้ามเขียนคำตอบใหม่เอง ให้รายงานผลการตรวจเท่านั้น (ถ้า read_sources ได้ "empty" และร่างมีข้อเท็จจริง ให้ตอบ ISSUES)"""


def build_agents(
    kb: KnowledgeBase,
    model="gemini-2.5-flash",
    *,
    researcher_model=None,
    checker_model=None,
) -> LlmAgent:
    """Wire the agents together. Each model can be a name or a model object (tests pass fakes)."""
    search_thai_wikipedia, read_sources = make_tools(kb)

    researcher = LlmAgent(
        name="researcher",
        model=researcher_model or model,
        description="Finds facts about Thai places in Thai Wikipedia and reports them with reference ids. Give it one specific sub-question naming the place.",
        instruction=RESEARCHER_INSTRUCTION,
        tools=[search_thai_wikipedia],
    )
    fact_checker = LlmAgent(
        name="fact_checker",
        model=checker_model or model,
        description="Checks a draft answer against the collected source passages. Give it the full draft with its reference ids; it replies OK or ISSUES.",
        instruction=CHECKER_INSTRUCTION,
        tools=[read_sources],
    )
    return LlmAgent(
        name="coordinator",
        model=model,
        description="Answers questions about Thai places from Thai Wikipedia by delegating to a researcher and a fact checker.",
        instruction=COORDINATOR_INSTRUCTION,
        tools=[AgentTool(researcher), AgentTool(fact_checker)],
    )
