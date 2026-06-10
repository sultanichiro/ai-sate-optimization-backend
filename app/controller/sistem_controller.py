from app.services.qlearning_service import QLearning

class SistemController:

    def __init__(self, db, db_manager, pedagang_id):
        self.db = db
        self.db_manager = db_manager
        self.pedagang_id = pedagang_id

    def jalankan_q_learning(self, data):

        ql = QLearning(states=1000, actions=2)

        episodes = {}

        for d in data:
            hari = d.hari_kuliah

            if hari not in episodes:
                episodes[hari] = []

            episodes[hari].append(d)

        total_reward = 0
        episode_rewards = []

        for hari, rows in episodes.items():

            ep_reward = 0

            for i in range(len(rows) - 1):

                state = (
                    rows[i].id_lokasi,
                    rows[i].kondisi_cuaca,
                    rows[i].hari_kuliah
                )

                next_state = (
                    rows[i+1].id_lokasi,
                    rows[i+1].kondisi_cuaca,
                    rows[i+1].hari_kuliah
                )

                action = ql.choose_action(state)

                reward = rows[i].jumlah_terjual

                ep_reward += reward

                ql.update(state, action, reward, next_state)

            episode_rewards.append(ep_reward)
            total_reward += ep_reward

        return {
            "success": True,
            "message": "Q-Learning selesai",
            "q_table": ql.q_table,
            "total_reward": total_reward,
            "episode_rewards": episode_rewards,
            "rekomendasi": "Pilih lokasi dengan Q-value tertinggi",
            "rute_optimal": []
        }