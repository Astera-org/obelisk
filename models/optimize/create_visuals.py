import wandb
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class visualize_bones():
    def __init__(self,apikey, runid):
        wandb.login(key = apikey)
        api = wandb.Api()

        self.apikey = apikey
        self.data_of_interest = None
        self.run = api.run(runid)
        self.score_key = f"best_observation/output"
        self.create_data()

    def create_data(self):
        history_df = self.run.history()
        history_df = history_df.replace("Infinity", float("-inf"))
        self.data_of_interest = history_df

    def create_best_figure(self):
        is_best_shown = True
        is_search_space_shown = True
        history_df = self.data_of_interest
        sns.set(rc={"figure.figsize": (12, 4)})
        sns.set_theme(style="whitegrid")
        cmap = sns.color_palette("crest", as_cmap=True)
        output_observation_df = history_df[["observation_count", f"observation/output"]].dropna()
        observation_x, observation_y = output_observation_df.to_numpy().T

        if is_search_space_shown:
            mean_std_df = history_df[["observation_count", "search_space_mean", "search_space_std_dev"]].dropna()
            search_space_x, search_space_mean, search_space_std = mean_std_df.to_numpy().T
            plt.plot(search_space_x, search_space_mean, color="grey", linewidth=2, label="search space mean")

            plt.fill_between(
                search_space_x,
                search_space_mean - search_space_std,
                search_space_mean + search_space_std,
                color="grey",
                alpha=0.1,
                label="search space",
                )

        if is_best_shown:
            output_best_observation_df = history_df[["observation_count", self.score_key]].dropna()
            best_observation_x, best_observation_y = output_best_observation_df.to_numpy().T
            plt.plot(best_observation_x, best_observation_y, linestyle="dashed", linewidth=2, label="best single observation")
        plt.scatter(observation_x, observation_y, c=-np.clip(observation_y, 0.8, 1), s=20, label="observation", cmap=cmap)
        plt.title("Combined performance metric")
        plt.xlabel("Observation count")
        plt.ylabel("Performance")
        ymin = history_df[self.score_key].dropna().min()
        ymax = history_df[self.score_key].dropna().max()
        plt.ylim(ymin, ymax)
        plt.legend()


    def write_best_local(self, path):
        self.create_data()
        self.create_best_figure()
        plt.savefig(path)
        plt.show()
