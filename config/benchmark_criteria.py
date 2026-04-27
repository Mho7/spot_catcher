"""
PatchCore 추론 속도 등급 임계값 (HCI 학술 자료 기반).

근거 (APA 형식):

Bouch, A., Kuchinsky, A., & Bhatti, N. (2000). Quality is in the eye of the
    beholder: Meeting users' requirements for Internet quality of service.
    Proceedings of the SIGCHI Conference on Human Factors in Computing Systems
    (CHI 2000), 297-304.

Miller, R. B. (1968). Response time in man-computer conversational
    transactions. AFIPS Fall Joint Computer Conference, 33, 267-277.

Nah, F. F.-H. (2004). A study on tolerable waiting time: How long are
    Web users willing to wait? Behaviour & Information Technology,
    23(3), 153-163.

Shneiderman, B. (1984). Response time and display rate in human performance
    with computers. ACM Computing Surveys, 16(3), 265-285.
"""

THRESHOLD_GOOD_MS = 7000      # 양호: t <= 7초
                              # 근거: 본 시스템 baseline 성능, Nah (2004) TWT 5-8초 범위 내
THRESHOLD_CAUTION_MS = 10000  # 주의: 7 < t <= 10초
                              # 근거: Miller (1968) / Shneiderman (1984) 주의력 한계
                              # 나쁨: t > 10초 (주의력 한계 초과)


def classify_latency(latency_ms: float) -> str:
    """추론 latency(ms)를 양호 / 주의 / 나쁨 3단계로 분류."""
    if latency_ms <= THRESHOLD_GOOD_MS:
        return "양호"
    elif latency_ms <= THRESHOLD_CAUTION_MS:
        return "주의"
    else:
        return "나쁨"
