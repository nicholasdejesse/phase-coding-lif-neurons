from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import argparse

def get_loss_from_events(log_dir, tag_name="Loss/test"):
    # Load the event files from your log folder
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()
    print(event_accumulator.Tags()["scalars"])
    # Grab the specific events by tag (e.g., training loss)
    events = event_accumulator.Scalars(tag_name)
    
    # Extract just the numerical value from each event
    loss_values = [event.value for event in events]
    return loss_values

def main():
    parser = argparse.ArgumentParser(description="Plot loss from TensorBoard logs.")
    parser.add_argument("--log_dir", type=str, required=True, help="Path to the TensorBoard log directory.")
    parser.add_argument("--tag_name", type=str, default="Accuracy/test", help="Tag name for the loss to plot.")
    args = parser.parse_args()

    loss_values = get_loss_from_events(args.log_dir, args.tag_name)
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label=args.tag_name)
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(top=100) 
    plt.legend()

    plt.savefig("plot.png")

if __name__ == "__main__":
    main()