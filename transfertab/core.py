# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_transfer.ipynb (unless otherwise specified).

__all__ = ['TabTransfer']

# Cell
class TabTransfer:
    def __init__(self, new_learner):
        self.new_cat_names = new_learner.dls.cat_names
        self.new_all_classes = new_learner.dls.classes
        self.new_learner = new_learner

    def init_from_json(self, path):
        f = open(path, 'rb')
        json_file = json.load(f)
        self.old_cat_names = json_file['categories']
        classes_dict = dict.fromkeys(self.old_cat_names)
        embed_dict = dict.fromkeys(self.old_cat_names)

        for i in self.old_cat_names:
            classes_dict[i] = json_file[i]['classes']
            embed_dict[i] = json_file[i]['embeddings']
        self.old_all_classes = classes_dict # contains the cat variable with all its classes
        self.old_all_embeds = embed_dict # contains the cat variable with all its embeddings

    def transfer(self, cat_names_to_transfer, verbose=False):
        self.transfer_list = cat_names_to_transfer
        for curr_cat in self.transfer_list:
            if not (curr_cat in self.old_cat_names and curr_cat in self.new_cat_names):
                continue
            old_cat_idx = self.old_cat_names.index(curr_cat)
            new_cat_idx = self.new_cat_names.index(curr_cat)



            # TODO: Make it so that this isn't required by taking care of this.
            try: assert (len(tabobj.old_all_embeds[curr_cat][0]) == self.new_learner.model.embeds[new_cat_idx].embedding_dim)
            except:
                print(f"Encountered an error for variable {curr_cat}: Make sure embeddings dimensions are same for {self.old_all_embeds[curr_cat]} with size {len(tabobj.old_all_embeds[curr_cat])}, and {self.new_learner.model.embeds[new_cat_idx]} with size {self.new_learner.model.embeds[new_cat_idx].embedding_dim}")
                print("Moving on to other cat vars")
                continue

            old_curr_classes = self.old_all_classes[curr_cat]
            new_curr_classes = self.new_all_classes[curr_cat]

            torch_embeds = torch.FloatTensor(self.old_all_embeds[curr_cat])
            weights_mean = torch_embeds.mean(0)

            if verbose: print(f'mean is {weights_mean} for {torch_embeds}')

            # Case where some category in old, but not in new isn't being handled rn.

            for new_curr_class in new_curr_classes:
                new_curr_class_idx = new_curr_classes.o2i[new_curr_class]
                if verbose: print(f"{new_curr_class_idx}, {type(new_curr_class_idx)}")

                if new_curr_class in old_curr_classes:
                    old_curr_class_idx = old_curr_classes.index(new_curr_class)
                    if verbose: print(f'Transferring weights for class {new_curr_class}, cat {curr_cat} from previous weights')
                    if verbose: print(f"old weight for class is {self.new_learner.model.embeds[new_cat_idx].weight[new_curr_class_idx, :]}")
                    tempwgt1 = self.new_learner.model.embeds[new_cat_idx].weight[new_curr_class_idx, :]
                    tempwgt2 = torch_embeds[old_curr_class_idx, :]

                    self.new_learner.model.embeds[new_cat_idx].weight.data[new_curr_class_idx, :] = torch_embeds[old_curr_class_idx, :].detach().clone()
                    self.new_learner.model.embeds[new_cat_idx].weight[new_curr_class_idx, :].required_grad = True
                    if verbose: print(f"new weight for class is {self.new_learner.model.embeds[new_cat_idx].weight[new_curr_class_idx, :]}")
                else:
                    if verbose: print(f'Transferring weights for class {new_curr_class}, cat {curr_cat} using mean')
                    if verbose: print(f"old weight for class is {self.new_learner.model.embeds[new_cat_idx].weight[new_curr_class_idx, :]}")
                    self.new_learner.model.embeds[new_cat_idx].weight.data[new_curr_class_idx, :] = weights_mean
                    self.new_learner.model.embeds[new_cat_idx].weight[new_curr_class_idx, :].required_grad = True
                    if verbose: print(f"new weight for class is {self.new_learner.model.embeds[new_cat_idx].weight[new_curr_class_idx, :]}")