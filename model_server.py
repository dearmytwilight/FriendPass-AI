from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict

app = FastAPI()

# 1. 요청 바디 정의
class User(BaseModel):
    id: int
    region: str
    is_exchange: int
    interests: List[str]

# 그룹화
category_mapping = {
   "k-pop" : "예술",
    "맛집" : "음식/카페",
    "산책" : "체험",
    "쇼핑" : "쇼핑/패션",
    "역사" : "문화/역사",
    "이색문화체험" : "체험",
    "전시" : "예술",
    "전통문화" : "문화/역사",
    "카페" : "음식/카페",
    "패션" : "쇼핑/패션",
    "힐링" : "체험"
}

all_groups = list(set(category_mapping.values()))
regions = ["국민", "성신", "동덕"]

def recommend_teams(users: list, threshold: float = 0.55) -> list:
    if not users:
        return []

    # 관심사 그룹화
    for u in users:
        u["grouped_interests"] = [category_mapping.get(c) for c in u["interests"] if category_mapping.get(c)]

    # 원핫 인코딩
    mlb = MultiLabelBinarizer(classes=all_groups)
    interest_matrix = mlb.fit_transform([u["grouped_interests"] for u in users])

    # 유저 벡터 생성
    user_index_map = {u["id"]: idx for idx, u in enumerate(users)}
    user_vectors = {}
    for u in users:
        region_onehot = [int(u["region"] == r) for r in regions]
        user_vectors[u["id"]] = np.array([u["is_exchange"]] + region_onehot + interest_matrix[user_index_map[u["id"]]].tolist())

    # 지역별 후보 분리
    region_candidates = {}
    for r in regions:
        ex_users = [u for u in users if u["is_exchange"] == 1 and u["region"] == r]
        kr_users = [u for u in users if u["is_exchange"] == 0 and u["region"] == r]
        region_candidates[r] = (ex_users, kr_users)

    # 팀 조합 생성
    teams = []
    for r, (ex_users, kr_users) in region_candidates.items():
        if len(ex_users) >= 2 and len(kr_users) >= 2:
            teams += list(itertools.product(
                itertools.combinations(ex_users, 2),
                itertools.combinations(kr_users, 2)
            ))

    if not teams:
        return []

    # 팀 벡터 및 라벨 생성
    X, y, team_meta = [], [], []
    all_sims = []

    for ex_pair, kr_pair in teams:
        team_ids = [u["id"] for u in ex_pair + kr_pair]
        team_vector = np.concatenate([
            user_vectors[ex_pair[0]["id"]],
            user_vectors[ex_pair[1]["id"]],
            user_vectors[kr_pair[0]["id"]],
            user_vectors[kr_pair[1]["id"]]
        ])
        X.append(team_vector)

        team_interests = np.array([
            interest_matrix[user_index_map[u["id"]]] for u in ex_pair + kr_pair
        ])
        sim_matrix = cosine_similarity(team_interests)
        n = sim_matrix.shape[0]
        sim_avg = (sim_matrix.sum() - n) / (n * (n - 1))
        all_sims.append(sim_avg)

        # 디버깅 로그
        print(f"팀 {team_ids} sim_avg={sim_avg:.3f}")

        team_meta.append({
            "team_ids": team_ids,
            "sim_avg": sim_avg,
            "ex_pair": ex_pair,
            "kr_pair": kr_pair
        })

    # 중앙값 기준 라벨링 (0/1 섞이도록)
    median_sim = np.median(all_sims)
    for idx, tm in enumerate(team_meta):
        label = 1 if tm["sim_avg"] >= median_sim else 0
        y.append(label)

    print("팀 라벨 분포:", Counter(y))
    print("sim_avg 최소:", round(min(all_sims), 3))
    print("sim_avg 최대:", round(max(all_sims), 3))
    print("sim_avg 평균:", round(np.mean(all_sims), 3))

    # 후보팀이 1개거나 라벨이 1개뿐이면 threshold 기준으로 추천
    if len(set(y)) < 2:
        print("팀 라벨이 하나뿐이라 학습 불가 → threshold 기준 추천")
        scored_teams = [tm for tm in team_meta if tm["sim_avg"] >= threshold]
        # 점수 대신 sim_avg 사용
        for tm in scored_teams:
            tm["score"] = tm["sim_avg"]
    else :

        model = LogisticRegression()
        model.fit(np.array(X), np.array(y))
        scores = model.predict_proba(np.array(X))[:, 1]

        scored_teams = [
            {
                "team_ids": team_meta[i]["team_ids"],
                "score": round(scores[i], 3),
                "ex_pair": team_meta[i]["ex_pair"],
                "kr_pair": team_meta[i]["kr_pair"]
            }
            for i in range(len(scores)) if scores[i] >= threshold
        ]
    scored_teams.sort(key=lambda x: x["score"], reverse=True)

    # 중복 유저 제거하며 확정
    used_ids = set()
    final_teams = []
    for team in scored_teams:
        if all(uid not in used_ids for uid in team["team_ids"]):
            team_raw_interests = list(itertools.chain.from_iterable([u["interests"] for u in team["ex_pair"] + team["kr_pair"]]))
            team_grouped_interests = list(itertools.chain.from_iterable([u["grouped_interests"] for u in team["ex_pair"] + team["kr_pair"]]))

            top_3_groups = [g for g, _ in Counter(team_grouped_interests).most_common(3)]
            group_to_original = defaultdict(list)
            for interest in team_raw_interests:
                group = category_mapping.get(interest)
                if group in top_3_groups:
                    group_to_original[group].append(interest)

            representatives_interests = []
            for group in top_3_groups:
                originals = group_to_original[group]
                if originals:
                    most_common = Counter(originals).most_common(1)[0][0]
                    representatives_interests.append(most_common)

            team_region = team["ex_pair"][0]["region"]

            final_teams.append({
                "team_ids": team["team_ids"],
                "score": float(team["score"]),
                "representative_interests": representatives_interests,
                "matched_region": team_region
            })
            used_ids.update(team["team_ids"])

    print("최종 응답:", final_teams)
    return final_teams



@app.post("/recommend-teams")
def recommend_endpoint(users: List[User]):
    print("유저 수:", len(users))
    print("첫 번째 유저:", users[0])
    

    users_dict = [u.dict() for u in users]
    result = recommend_teams(users_dict)
    return result

# 4. 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model_server:app", host="0.0.0.0", port=8000)