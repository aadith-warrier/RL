from matplotlib import animation
import matplotlib.pyplot as plt

def define_env_parameters(env):
    action_space = env.single_action_space
    obs_space = env.single_observation_space
    env_parameters = {
        "action_space":action_space,
        "obs_space":obs_space
    }
    return env_parameters

def save_frames_as_gif(frames, path='./', filename='top_of_the_world.gif'):

    #A utility function to save the succesful episodes as gifs. 
    # Taken from: 

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def plot_reward_epsiode():
    pass

def plot_timesteps_episodes():
    pass

def save_model():
    pass

def load_model():
    pass