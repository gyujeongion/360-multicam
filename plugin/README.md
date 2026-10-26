# orbcut — 프리미어 UXP 패널

orbcut 결과(`premiere.xml`, `premiere_shots.json`)를 프리미어 안에서 다룬다. 처리(스티칭·화자 감지·마스터 렌더)는 `orbcut` CLI 담당.

## 설치 (개발용)
1. Adobe **UXP Developer Tool** 실행 → *Add Plugin* → 이 폴더의 `manifest.json` 선택
2. 프리미어(25.6 이상) 실행 상태에서 *Load* → 프리미어 메뉴 *창 > UXP 플러그인 > orbcut*

## 사용
1. **XML 가져오기**로 `premiere.xml` (또는 파일 > 가져오기) → 생긴 시퀀스 열기
2. **결과 불러오기**로 같은 폴더의 `premiere_shots.json`
3. **자동 구도 전체 적용** — V1 클립마다 얼굴 기준 Scale/Position (XML 모션이 이미 맞으면 생략 가능)
4. 잘린 샷은 클립 선택 후 **확대/축소/위/아래/왼쪽/오른쪽**으로 미세조정 (재렌더 없음)
5. 샷 목록 클릭 = 그 샷 시작으로 이동

## 실기 확인 (Premiere Pro 26.5, 2026-10-22)
- 설치: UXP Developer Tool 없이도 `.ccx`(이 폴더 4개 파일 zip)를 Creative Cloud의 `UnifiedPluginInstallerAgent --install x.ccx`로 설치 가능 → 창 > UXP 플러그인 > orbcut
- 결과 불러오기: 활성 시퀀스 V1 클립의 미디어(`<out>/masters/cam_X.mp4`) 위치에서 `premiere_shots.json`을 자동으로 찾음(파일 선택 불필요, 실패 시 선택창)
- 자동 구도 적용: Position = 0.5 + horiz/vert (0~1 정규화 PointF) — 프리뷰와 픽셀 대조 잔차 ≤1.1px
- 액션·키프레임은 `lockedAccess` 안에서 만들어야 함(밖에서 만들면 "script object is no longer valid")

검증: `node plugin/scripts/validate.mjs` (매니페스트·구문·API 이름 문서 대조)
