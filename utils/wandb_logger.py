import wandb


def InitWandb(project_name, group_name, run_name, parameters):
    print(f"initializing wandb project {project_name}, group {group_name}, run {run_name}...")
    wandb.init(
        project=project_name,
        group=group_name,
        name=run_name,
        config = parameters
    )

def LoggingWandb(cur_acc_loss, epoch, optimizer):
    print(f"logging to epoch {epoch} in wandb...")
    log_dict = {}
    log_dict["epoch"] = epoch
    log_dict["lr"] = optimizer.param_groups[0]["lr"]
    for num_bis, (clean_acc, clean_loss, backdoor_acc, backdoor_loss) in cur_acc_loss.items(): 
        log_dict[f"{num_bis}_bits/clean_acc"] = clean_acc
        log_dict[f"{num_bis}_bits/clean_loss"] = clean_loss
        log_dict[f"{num_bis}_bits/backdoor_acc"] = backdoor_acc
        log_dict[f"{num_bis}_bits/backdoor_loss"] = backdoor_loss
    wandb.log(log_dict, step=epoch)