import time

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader


class TSVS_Dataset:
    def __init__(self, ):
        pass

    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return f'svs{str(i)}'

    def __len__(self):
        return 200


class TSVC_Dataset:
    def __init__(self, ):
        pass

    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return f'svc{str(i)}'

    def __len__(self):
        return 150


class MIX_Dataset(Dataset):
    def __init__(self, svs_data_set, svc_data_set):
        self.svs_data_set = svs_data_set
        self.svc_data_set = svc_data_set
        self.num_for_svs = len(self.svs_data_set)
        self.num_for_svc = len(self.svc_data_set)
        self.mix_num = self.num_for_svs + self.num_for_svc

    def __getitem__(self, i):

        if i < self.num_for_svs:
            return self.svs_data_set[i]
        else:
            return self.svc_data_set[i - self.num_for_svs]

    def __len__(self):
        return self.mix_num

    def get_child_data_set_num(self):
        return {'svs': self.num_for_svs, 'svc': self.num_for_svc}


class ssvvsc_BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, svs_batch_size=None,
                 num_replicas=None, rank=None,

                 shuffle=True, seed=0, drop_last=False) -> None:

        if svs_batch_size is None:
            svs_batch_size = batch_size // 2

        self.svs_batch_size = svs_batch_size
        self.svc_batch_size = batch_size - svs_batch_size
        assert svs_batch_size < batch_size

        assert svs_batch_size > 0

        self.dataset = dataset
        nums = self.dataset.get_child_data_set_num()
        self.num_for_svs = nums['svs']
        self.num_for_svc = nums['svc']

        self.batch_size = batch_size

        self.num_replicas = num_replicas
        self.rank = rank

        self.shuffle = shuffle

        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.isfirst = True
        # t1=time.time()
        self.bls = self.build_batch()
        # t2 = time.time()
        # print(t2-t1)
        self.blsn = len(self.bls)

    def set_epoch(self,btch):
        self.epoch =btch

    # def listcut(self,Llist)->list:
    #     ctl=[]
    #     for i in range(self.num_replicas):
    #         ctl.append([].copy())
    #
    #     for idx,i in enumerate(Llist):
    #         ctl[idx%self.num_replicas].append(i)
    #     return ctl

    def build_batch(self):
        svs_npad = self.num_for_svs % self.svs_batch_size
        if svs_npad != 0:
            svs_pad = self.svs_batch_size - svs_npad

        svc_npad = self.num_for_svc % self.svc_batch_size
        if svc_npad != 0:
            svc_pad = self.svc_batch_size - svc_npad

        if self.drop_last:
            svs_bnum = self.num_for_svs // self.svs_batch_size
            svc_bnum = self.num_for_svc // self.svc_batch_size
        else:
            svs_bnum = self.num_for_svs // self.svs_batch_size
            svc_bnum = self.num_for_svc // self.svc_batch_size
            if svs_npad != 0:
                svs_bnum = svs_bnum + 1
            if svc_npad != 0:
                svc_bnum = svc_bnum + 1

        batchlist = []  # svc 数据前置

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if svs_bnum > svc_bnum:
            if self.shuffle :
                svsidl = torch.randperm(self.num_for_svs, generator=g).tolist()
            else:
                svsidl = list(range(self.num_for_svs))
            tpmlist = []
            tpmlist2 = []
            if self.drop_last and svs_npad != 0:
                svsidl = svsidl[:-svs_npad]
            else:
                if svs_npad != 0:
                    if self.shuffle:
                        svsidl = svsidl + torch.randperm(self.num_for_svs, generator=g).tolist()[:svs_pad]
                    else:
                        svsidl = svsidl + list(range(self.num_for_svs))[:svs_pad]
            sxcl = []
            for i in svsidl:
                if len(sxcl) == self.svs_batch_size:
                    tpmlist.append(sxcl.copy())
                    sxcl = []
                sxcl.append(i)
            tpmlist.append(sxcl.copy())

            svsblen = len(tpmlist)

            svcnumitem = svsblen * self.svc_batch_size

            itspad = svcnumitem % self.num_for_svc
            itsnpad = self.num_for_svc - itspad
            if itspad != 0:
                svccpac = svcnumitem // self.num_for_svc + 1
            else:
                svccpac = svcnumitem // self.num_for_svc

            svcidl = []

            for i in range(svccpac):
                if self.shuffle:
                    svcidl = svcidl + (torch.randperm(self.num_for_svc, generator=g) + self.num_for_svs).tolist()
                else:
                    svcidl = svcidl + (torch.tensor(list(range(self.num_for_svc))) + self.num_for_svs).tolist()

            if itspad != 0:
                svcidl = svcidl[:-itsnpad]
            svc_ctpm = []
            for i in svcidl:
                if len(svc_ctpm) == self.svc_batch_size:
                    tpmlist2.append(svc_ctpm.copy())
                    svc_ctpm = []
                svc_ctpm.append(i)
            tpmlist2.append(svc_ctpm.copy())

            assert len(tpmlist) == len(tpmlist2)

            for svcX, svsX in zip(tpmlist, tpmlist2):
                svcX: list
                svsX: list
                batchlist.append(svcX + svsX)
        elif svs_bnum < svc_bnum:
            if self.shuffle:
                svcidl = (torch.randperm(self.num_for_svc, generator=g) + self.num_for_svs).tolist()
            else:
                svcidl = (torch.tensor(list(range(self.num_for_svc))) + self.num_for_svs).tolist()
            tpmlist = []
            tpmlist2 = []
            if self.drop_last and svc_npad != 0:
                svcidl = svcidl[:-svc_npad]
            else:
                if svc_npad != 0:
                    if self.shuffle:
                        svcidl = svcidl + (torch.randperm(self.num_for_svc, generator=g) + self.num_for_svs).tolist()[
                                          :svc_pad]
                    else:
                        svcidl = svcidl + (torch.tensor(list(range(self.num_for_svc))) + self.num_for_svs).tolist()[
                                          :svc_pad]
            sxcl = []
            for i in svcidl:
                if len(sxcl) == self.svc_batch_size:
                    tpmlist.append(sxcl.copy())
                    sxcl = []
                sxcl.append(i)
            tpmlist.append(sxcl.copy())

            svcblen = len(tpmlist)

            svsnumitem = svcblen * self.svs_batch_size

            itspad = svsnumitem % self.num_for_svs
            itsnpad = self.num_for_svs - itspad
            if itspad != 0:
                svscpac = svsnumitem // self.num_for_svs + 1
            else:
                svscpac = svsnumitem // self.num_for_svs

            svsidl = []

            for i in range(svscpac):
                if self.shuffle:
                    svsidl = svsidl + (torch.randperm(self.num_for_svs, generator=g)).tolist()
                else:
                    svsidl = svsidl + list(range(self.num_for_svs))

            if itspad != 0:
                svsidl = svsidl[:-itsnpad]
            svc_ctpm = []
            for i in svsidl:
                if len(svc_ctpm) == self.svs_batch_size:
                    tpmlist2.append(svc_ctpm.copy())
                    svc_ctpm = []
                svc_ctpm.append(i)
            tpmlist2.append(svc_ctpm.copy())

            assert len(tpmlist) == len(tpmlist2)

            for svsX, svcX in zip(tpmlist, tpmlist2):
                svcX: list
                svsX: list
                batchlist.append(svcX + svsX)
        else:
            if self.shuffle :
                svsidl = torch.randperm(self.num_for_svs, generator=g).tolist()
            else:
                svsidl = list(range(self.num_for_svs))
            tpmlist = []
            tpmlist2 = []
            if self.drop_last and svs_npad != 0:
                svsidl = svsidl[:-svs_npad]
            else:
                if svs_npad != 0:
                    if self.shuffle:
                        svsidl = svsidl + torch.randperm(self.num_for_svs, generator=g).tolist()[:svs_pad]
                    else:
                        svsidl = svsidl + list(range(self.num_for_svs))[:svs_pad]
            sxcl = []
            for i in svsidl:
                if len(sxcl) == self.svs_batch_size:
                    tpmlist.append(sxcl.copy())
                    sxcl = []
                sxcl.append(i)
            tpmlist.append(sxcl.copy())

            svsblen = len(tpmlist)

            svcnumitem = svsblen * self.svc_batch_size

            itspad = svcnumitem % self.num_for_svc
            itsnpad = self.num_for_svc - itspad
            if itspad != 0:
                svccpac = svcnumitem // self.num_for_svc + 1
            else:
                svccpac = svcnumitem // self.num_for_svc

            svcidl = []

            for i in range(svccpac):
                if self.shuffle:
                    svcidl = svcidl + (torch.randperm(self.num_for_svc, generator=g) + self.num_for_svs).tolist()
                else:
                    svcidl = svcidl + (torch.tensor(list(range(self.num_for_svc))) + self.num_for_svs).tolist()

            if itspad != 0:
                svcidl = svcidl[:-itsnpad]
            svc_ctpm = []
            for i in svcidl:
                if len(svc_ctpm) == self.svc_batch_size:
                    tpmlist2.append(svc_ctpm.copy())
                    svc_ctpm = []
                svc_ctpm.append(i)
            tpmlist2.append(svc_ctpm.copy())

            assert len(tpmlist) == len(tpmlist2)

            for svcX, svsX in zip(tpmlist, tpmlist2):
                svcX: list
                svsX: list
                batchlist.append(svcX + svsX)

        if self.num_replicas > 1:
            nbs = len(batchlist)
            if nbs % self.num_replicas != 0:
                adx = []
                padl = torch.randperm(nbs, generator=g).tolist()[:self.num_replicas - (nbs % self.num_replicas)]
                for j in padl:
                    adx.append(batchlist[j].copy())
                batchlist += adx
            nbr = nbs // self.num_replicas

            # cutl=self.listcut(batchlist)

            return batchlist[self.rank * nbr:(self.rank + 1) * nbr]
        else:
            return batchlist

    def __iter__(self):
        if not self.isfirst:
            # t1 = time.time()
            self.bls = self.build_batch()

            # t2 = time.time()
            # print(t2 - t1)
        self.isfirst = False
        return iter(self.bls)

    def __len__(self) -> int:
        return self.blsn


class ssvvsc_BatchSampler_val(Sampler):
    def __init__(self, dataset,
                 num_replicas=None, rank=None,

                 ) -> None:





        self.dataset = dataset
        nums = self.dataset.get_child_data_set_num()
        self.num_for_svs = nums['svs']
        self.num_for_svc = nums['svc']


        self.num_replicas = num_replicas
        self.rank = rank



        self.epoch = 0
        self.bls=[]
        for i in range(len(self.dataset)):
            self.bls.append([i])


        self.blsn = len(self.bls)

    def set_epoch(self,btch):
        self.epoch =btch

    # def listcut(self,Llist)->list:
    #     ctl=[]
    #     for i in range(self.num_replicas):
    #         ctl.append([].copy())
    #
    #     for idx,i in enumerate(Llist):
    #         ctl[idx%self.num_replicas].append(i)
    #     return ctl



    def __iter__(self):


            # t2 = time.time()
            # print(t2 - t1)
        if self.rank is not None and self.rank!=0:
            return iter([[0]])

        return iter(self.bls)

    def __len__(self) -> int:
        return self.blsn




if __name__ == '__main__':
    from tqdm import tqdm

    mixd = MIX_Dataset(svs_data_set=TSVS_Dataset(), svc_data_set=TSVC_Dataset())
    ddl = torch.utils.data.DataLoader(mixd,
                                      # collate_fn=self.train_dataset.collater,
                                      batch_sampler=ssvvsc_BatchSampler(mixd, batch_size=5, svs_batch_size=2,
                                                                        num_replicas=1, rank=None,

                                                                        shuffle=True, seed=0, drop_last=False),
                                      # num_workers=hparams['ds_workers'],
                                      # prefetch_factor=hparams['dataloader_prefetch_factor'],
                                      # pin_memory=True,
                                      # persistent_workers=True

                                      )

    for i in tqdm(ddl):
        print(i)
        pass
    print('\n')
    ddl.batch_sampler.update_epoch(2)
    for i in tqdm(ddl):
        # print(i)
        pass
    print('\n')
    ddl.batch_sampler.update_epoch(3)
    for i in tqdm(ddl):
        # print(i)
        pass
