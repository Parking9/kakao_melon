import os
import pandas as pd
import numpy as np
import warnings
import json
import datetime
import copy
import random
from collections import Counter
from gensim.models import Word2Vec
import re
import sentencepiece as spm
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

os.chdir('C:\\Users\\ie gram_26\\Desktop\\7th')
json_data1=open('train.json',encoding='UTF-8').read()
train = json.loads(json_data1)

json_data2=open('test.json',encoding='UTF-8').read()
test = json.loads(json_data2)

# json_data3=open('val.json',encoding='UTF-8').read()
# val_df = json.loads(json_data3)

json_data4=open('song_meta.json',encoding='UTF-8').read()
song_meta = json.loads(json_data4)

genre_gn_all = pd.read_json('genre_gn_all.json', typ = 'series',encoding='UTF-8')

# 장르코드 : gnr_code, 장르명 : gnr_name
genre_df = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})
train_columns=list(train[0].keys())
test_columns=list(test[0].keys())
song_columns=list(song_meta[0].keys())
# val_columns=list(val_df[0].keys())
train_columns=['id','plylst_title','tags','songs','like_cnt','updt_date']
test_columns=['id','plylst_title','tags','songs','like_cnt','updt_date']
train=pd.DataFrame(train,columns=train_columns)
test=pd.DataFrame(test,columns=test_columns)
song_meta=pd.DataFrame(song_meta)
# val_df=pd.DataFrame(val_df)

# 곡 아이디(id)와 대분류 장르코드 리스트(song_gn_gnr_basket) 추출
song_gnr_map = song_meta.loc[:, ['id', 'song_gn_gnr_basket']]

# unnest song_gn_gnr_basket
song_gnr_map_unnest = np.dstack(
    (
        np.repeat(song_gnr_map.id.values, list(map(len, song_gnr_map.song_gn_gnr_basket))), 
        np.concatenate(song_gnr_map.song_gn_gnr_basket.values)
    )
)

# unnested 데이터프레임 생성 : song_gnr_map
song_gnr_map = pd.DataFrame(data = song_gnr_map_unnest[0], columns = song_gnr_map.columns)
song_gnr_map['id'] = song_gnr_map['id'].astype(str)
song_gnr_map.rename(columns = {'id' : 'song_id', 'song_gn_gnr_basket' : 'gnr_code'}, inplace = True)

# unnest 객체 제거
del song_gnr_map_unnest

# 1. 곡 별 장르 개수 count 테이블 생성 : song_gnr_count
song_gnr_count = song_gnr_map.groupby('song_id').gnr_code.nunique().reset_index(name = 'mapping_gnr_cnt')

# 2. 1번에서 생성한 테이블을 가지고 매핑된 장르 개수 별 곡 수 count 테이블 생성 : gnr_song_count
gnr_song_count = song_gnr_count.groupby('mapping_gnr_cnt').song_id.nunique().reset_index(name = '매핑된 곡 수')

# 3. 2번 테이블에 비율 값 추가
gnr_song_count.loc[:, '비율(%)'] = round(gnr_song_count['매핑된 곡 수']/sum(gnr_song_count['매핑된 곡 수'])*100, 2)
gnr_song_count = gnr_song_count.reset_index().rename(columns = {'mapping_gnr_cnt' : '장르 수'})
gnr_song_count[['장르 수', '매핑된 곡 수', '비율(%)']]
# 장르코드 뒷자리 두 자리가 00인 코드를 필터링
gnr_code = genre_df[genre_df['gnr_code'].str[-2:] == '00']
# 1. 장르 별 곡 수 count 테이블 생성 : gnr_count
gnr_count = song_gnr_map.groupby('gnr_code').song_id.nunique().reset_index(name = 'song_cnt')

# 2. 1번 테이블과 장르 meta와 join
gnr_count = pd.merge(gnr_count, gnr_code.loc[:, ['gnr_code', 'gnr_name']], how = 'left', on = 'gnr_code')
gnr_count['gnr_code_name'] = gnr_count['gnr_code'] + ' (' + gnr_count['gnr_name'] + ')'

# 3. 매핑이 되지 않은 일부 곡들은 제거
gnr_count = gnr_count[['gnr_code_name', 'song_cnt']].dropna()

# 4. 많은 곡이 매핑된 순 기준으로 내림차순 리스트 생성
gnr_list_desc = gnr_count.sort_values('song_cnt', ascending = False).gnr_code_name



train=train.astype({'id':'object'})
train['updt_date']=pd.to_datetime(train['updt_date'])

# 플레이리스트 아이디(id)와 수록곡(songs) 추출
plylst_song_map = train[['id', 'songs']]

# unnest songs
plylst_song_map_unnest = np.dstack(
    (
        np.repeat(plylst_song_map.id.values, list(map(len, plylst_song_map.songs))), 
        np.concatenate(plylst_song_map.songs.values)
    )
)

# unnested 데이터프레임 생성 : plylst_song_map
plylst_song_map = pd.DataFrame(data = plylst_song_map_unnest[0], columns = plylst_song_map.columns)
plylst_song_map['id'] = plylst_song_map['id'].astype(str)
plylst_song_map['songs'] = plylst_song_map['songs'].astype(str)

# unnest 객체 제거
del plylst_song_map_unnest

# 플레이리스트 아이디(id)와 매핑된 태그(tags) 추출
plylst_tag_map = train[['id', 'tags']]

# unnest tags
plylst_tag_map_unnest = np.dstack(
    (
        np.repeat(plylst_tag_map.id.values, list(map(len, plylst_tag_map.tags))), 
        np.concatenate(plylst_tag_map.tags.values)
    )
)

# unnested 데이터프레임 생성 : plylst_tag_map
plylst_tag_map = pd.DataFrame(data = plylst_tag_map_unnest[0], columns = plylst_tag_map.columns)
plylst_tag_map['id'] = plylst_tag_map['id'].astype(str)

# unnest 객체 제거
del plylst_tag_map_unnest

plyst=plylst_song_map.merge(plylst_tag_map,on='id')
like_c=train[['id','like_cnt']]
like_c['id']=like_c['id'].astype(str)
plyst=plyst.merge(like_c,on='id')
song_tag_like=plyst[['songs','tags','like_cnt']]

# 태그 별 매핑 빈도 수 저장 
tag_cnt = plylst_tag_map.groupby('tags').tags.count().reset_index(name = 'mapping_cnt')
tag_cnt['tags'] = tag_cnt['tags'].astype(str)
tag_cnt['mapping_cnt'] = tag_cnt['mapping_cnt'].astype(int)

# 빈도 수가 70회 이상인 태그만 저장
tag_cnt = tag_cnt[tag_cnt['mapping_cnt'] >= 70]
word_tag=copy.deepcopy(list(tag_cnt.tags))

#노래 별 태그 매핑 병합
plyst=plylst_tag_map.merge(plylst_song_map,on='id')
like_c=train[['id','like_cnt']]
like_c['id']=like_c['id'].astype(str)
#플레이리스트 좋아요 병합
plyst=plyst.merge(like_c,on='id')
tag70_song2=plyst[['songs','tags','like_cnt']]
#빈도 수 70회 이상 태그와 매핑된 노래
tag70_song2=tag70_song2[tag70_song2['tags'].apply(lambda x : x in word_tag)]

#같은 태그와 2회 이상 매핑된 노래
tag70_song2['count']=1
tag70_song2_1=tag70_song2[['songs','tags','count']]
rating=pd.pivot_table(tag70_song2_1,index='songs',columns='tags',aggfunc='sum').fillna(0)
rating=rating[rating.apply(lambda x: max(x)>=2,axis=1)]
rating_idx=rating.index
tag70_song2_1=tag70_song2[tag70_song2['songs'].apply(lambda x : x in rating_idx)][['songs','tags','like_cnt']]

#Row : song, Columns : tag, Value : like_cnt
rating=pd.pivot_table(tag70_song2_1,index='songs',columns='tags',aggfunc='sum').fillna(0)
rating_value=copy.deepcopy(list(rating.values))
rating_idx=copy.deepcopy(list(map(int,rating.index)))
tag_col=[]
for i in rating.columns:
    tag_col.append(i[1])
word_tag=copy.deepcopy(tag_col)



#song_meta 장르 이상치(GN9000) 처리, 가장 많이 플레이리스트에 함께 포함된 노래의 장르로 변경
meta2=song_meta.loc[list(song_meta.iloc[:,6].apply(lambda x: len(x))!=0),:] # meta에 장르가 []인 행 1059개 제거
# gnr이 0100 - 3000 사이만 추출 
meta2=meta2.loc[list(meta2['song_gn_gnr_basket'].apply(lambda x: int(x[0][2:4])>=1 and int(x[0][2:4])<=30)),:]
before=list(song_meta.index)
after=list(meta2.index)
gn_out=list(set(before) -set(after))


metadf=song_meta.loc[:,['song_gn_gnr_basket']]
metadf['cnt']=np.repeat(1,len(metadf))
metadf.iloc[:,0:1]=metadf.iloc[:,0].apply(tuple)    

dd=pd.DataFrame(None,index=gn_out,columns=['sub']) # 2893개에 대한 대체 테이블 생성
gn_out_tr=list(train.songs.apply(lambda x: len(list(set(x) & set(gn_out)))>0)) # 2893개가 포함된 행
train_ept9000=train.loc[gn_out_tr,'songs']
for i in gn_out:
    along=pd.DataFrame(train_ept9000[list(train_ept9000.apply(lambda x: i in x))])
    ll2=along.shape[0]

    ok_sub=[]
    for j in range(0,ll2):
        ok_sub=ok_sub+along.iloc[j,0]
    if list(set(ok_sub)-set([i])) !=[]:
        dd.loc[i,:]['sub']=((metadf.loc[list(set(ok_sub)-set([i])),:]).groupby('song_gn_gnr_basket')['cnt'].sum()).sort_values(ascending=False).index[0]
    else:
        pass

delist=list(dd.index[list(pd.isnull(dd['sub']))]) # 404개 null 삭제
gnsub=list(dd.loc[list(pd.notnull(dd['sub'])),:].index) # () + 온전한 2489개
dd2=dd.loc[gnsub,:] 
dgnsub=list(dd2.loc[list(pd.DataFrame(dd).loc[gnsub,:].applymap(lambda x:x!=())['sub']),:].index) # 온전한 2439개
delist=list(set(list(dd.index))-set(dgnsub))
song_meta.loc[dgnsub,'song_gn_gnr_basket']=pd.DataFrame(dd).loc[dgnsub,:].applymap(lambda x: list(x))['sub']


#song_meta 날짜 전처리, ex) 19870000, 20000332, 20100230 etc 
def date(x):
    if x[5:7]=='00':
        x=x[:5]+'01'+x[7:]
    if x[8:]=='00':
        x=x[:8]+'01'
    return x

song_meta['issue_date']=song_meta['issue_date'].apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:])
song_meta['issue_date']=song_meta['issue_date'].apply(date)
no_date=song_meta[song_meta['issue_date'].apply(lambda x: int(x[:4])==0)].index
song_meta.loc[no_date,'issue_date']=pd.Series(['1950-01-01' for i in range(len(no_date))],index=no_date)

monthout_date=song_meta[song_meta['issue_date'].apply(lambda x: int(x[5:7])>12)].index
song_meta.loc[monthout_date,'issue_date']=pd.Series(['2010-01-30'],index=monthout_date)

dayout_date=song_meta[song_meta['issue_date'].apply(lambda x: int(x[8:])>31)].index
song_meta.loc[dayout_date,'issue_date']=pd.Series(['2006-01-31'],index=dayout_date)

dayout_date2=song_meta[song_meta['issue_date'].apply(lambda x: int(x[8:])>30 and int(x[5:7]) in [2,4,6,9,11])].index
song_meta.loc[dayout_date2,'issue_date']=pd.Series(['2001-09-30'],index=dayout_date2)

song_meta['issue_date']=pd.to_datetime(song_meta['issue_date'])


#추출한 노래 발매 일 저장
song_4=song_meta[['id','issue_date']]
iss_date=[]
isd=list(song_4.issue_date)
ss=list(song_4.id)
l=len(rating_idx)
date_dic={}
for i in range(len(isd)):
    date_dic[ss[i]]=isd[i]
for i in rating_idx:
    iss_date.append(date_dic[i])
    
val_date=list(test.updt_date)


#추출한 노래끼리의 가까운 이웃 100개
X=rating.values
nbrs =NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
distances=list(distances)
indices=list(indices)

#Word2Vec 태그 간 유사도 학습, train 태그
length_tags = train.tags.shape[0]
total=[]
a=train.tags
for i in range(length_tags):

    total.append(a[i])
total= [x for x in total if len(x)>1]
#total

model = Word2Vec(total, min_count=3,size=100,window=210,sg=10) #최소 매핑 카운트 3회
voca=list(model.wv.vocab) #학습된 단어 vocab


#SentencePiece 태그와 노래가 모두 없을 때, 플레이리스트 제목에서 유의미한 단어 추출 
#train plylst_title, song_meta artist, gnr_name 학습 
join_voca='\n'.join(voca)
list_ply=list(train['plylst_title'])
join_ply='\n'.join(list_ply)

drop_artist=song_meta.dropna(subset=['artist_name_basket'])
list_artist=list(drop_artist['artist_name_basket'])
artist_list=[]
for i in range(len(list_artist)):
    for j in list_artist[i]:
        artist_list.append(j)
artist_list=list(set(artist_list))

join_artist='\n'.join(artist_list)

gnr_00=list(set(gnr_code.gnr_name.values))
join_gnr='\n'.join(gnr_00)

f=open('test.txt', mode='wt', encoding='utf-8')
f.write(join_ply)
f.write(join_artist)
f.write(join_voca)
f.write(join_gnr)
parameter = '--input={} --model_prefix={} --vocab_size={} --model_type={}'
input_file = 'test.txt'
vocab_size = 55000
prefix = 'bert_kor'
model_type = 'bpe'
cmd = parameter.format(input_file, prefix, vocab_size, model_type)
spm.SentencePieceTrainer.Train(cmd)

sp=spm.SentencePieceProcessor()
sp.Load('bert_kor.model')

ply_val=test[(test['tags'] + test['songs']).map(len) == 0]
ply_index=copy.deepcopy(ply_val.index)
list_title=list(ply_val['plylst_title'])
title=[]
length_4=len(ply_val)
for i in range(length_4):
    text=list_title[i]
    pattern=r'[ㄱ-ㅎ]'
    text=re.sub(pattern,r'',text)
    pattern=r'[^\w\s]'
    text=re.sub(pattern,r'',text)
    pattern=r'[ ]{2,}'
    text=re.sub(pattern,r'',text)
    pattern=r'[\u3000]+'
    text=re.sub(pattern,r'',text)
    pattern=r'[을,를]'
    text=re.sub(pattern,r'',text)
    title.append(text)
ply_token=[]
for i in range(length_4):
    k=sp.EncodeAsPieces(title[i])
    ply_token.append(k)
    

x=[]
for i in range(length_4):
    for k in range(len(ply_token[i])):
        x=ply_token[i][k].strip('▁')
        ply_token[i][k]=ply_token[i][k].replace(ply_token[i][k],x)     
ply_token_s=pd.Series(ply_token,index=ply_index)

#장르별 가장 많이 매핑된 노래 추출
song_gr=list(song_meta.song_gn_dtl_gnr_basket)
song_c=song_meta.loc[:,['id','song_gn_dtl_gnr_basket','song_gn_gnr_basket']]
song_count=Counter(tag70_song2_1.songs)
song_count=pd.DataFrame(song_count.values(),index=song_count.keys(),columns=['count'])
song_c=song_c.loc[list(map(int,list(song_count.index))),:].iloc[:,1:3]
song_c['count']=song_count.values

r_song=list(map(int,rating.index))

rt_count=song_c.loc[r_song,['song_gn_gnr_basket','count']]
rt_count.iloc[:,0:1]=rt_count.iloc[:,0:1].applymap(lambda x: tuple(x))
rt_count2=rt_count.sort_values(by='count',ascending=False)
rt_count2=rt_count2.groupby('song_gn_gnr_basket').head(10)
rt_count2_idx=rt_count2.song_gn_gnr_basket
rt_count2_value=rt_count2['count'].values
rt_count3=pd.Series(rt_count2_value,index=rt_count2_idx)
a2=list(rt_count3.index)
b2=list(rt_count3)

gn_max_song=len(rt_count3)
v=[]
for i in range(0,gn_max_song):
    v.append(list(set(list(np.where(rt_count['song_gn_gnr_basket']==a2[i])[0])) & set(list(np.where(rt_count['count']==b2[i])[0])))[0])
rt_count4=pd.DataFrame(rt_count3)
rt_count4['song']=list(rt_count.iloc[v,:].index)
rt_count4.reset_index(inplace=True)
rt_count4.columns=['song_gn_gnr_basket','count','song']

# 리스트 안의 모든 노래가 추출한 노래(raitng)에 없을 경우, 같은 장르에서 대체 노래 추출 
def no_sub(X,y,song_meta,song_gr,song_c,v_date,rating_idx):
    sub=pd.DataFrame(None,index=X,columns=['candi']).sort_index()
    for j in X:
        gn=song_gr[j]
        dt=v_date
        at=song_meta.loc[song_meta.id==j,:].iloc[0,4]  ##아티스트 일련번호
        
        candi=song_meta.loc[(song_meta.iloc[:,0].apply(lambda x: set(x)==set(gn)))]
        candi=candi.loc[(candi.iloc[:,1]<=dt)]
        candi=candi.loc[(candi.iloc[:,4].apply(lambda x: set(x)==set(at)))]
        
        sub.loc[j,'candi']=list(candi.index)
    
    rt_s=rating_idx

    sub['not_in_rt']=None
    for j in list(sub.index):
        sub['not_in_rt'][j]=list(set(sub.loc[j,'candi']) - set(rt_s))
        
    sub['in_rt']=None
    for j in list(sub.index):
        sub['in_rt'][j]=list(set(sub.loc[j,'candi']) - set(sub.loc[j,'not_in_rt']))
    ans=[]
    l1=len(sub)
    for j in range(0,l1):
        if sub.iloc[j,2:3][0] !=[]:
           ans.append((song_c.loc[sub.iloc[j,2:3][0],'count'].sort_values(ascending=False)).index[0])
        else:
            not_rt=list(song_c['song_gn_dtl_gnr_basket'].apply(lambda x: set(x)==set(song_meta.loc[sub.index[j],'song_gn_dtl_gnr_basket'])))
            if sum(not_rt)==0:
                not_rt2=list(song_c['song_gn_dtl_gnr_basket'].apply(lambda x: len(set(x)&set(song_meta.loc[sub.index[j],'song_gn_dtl_gnr_basket']))>1))
                if sum(not_rt2)==0:
                    not_rt3=list(song_c['song_gn_gnr_basket'].apply(lambda x: set(x)==set(song_meta.loc[sub.index[j],'song_gn_gnr_basket'])))
                    if sum(not_rt3)==0:
                        return 0
                    else:
                        ans.append(song_c.loc[not_rt3,'count'].sort_values(ascending=False).index[0])
                else:
                    ans.append(song_c.loc[not_rt2,'count'].sort_values(ascending=False).index[0])        
            else:
                ans.append(song_c.loc[not_rt,'count'].sort_values(ascending=False).index[0])
    ans=list(set(ans))
    return ans

#가까운 이웃 100개 중 중복되는 노래 추출, 거리 오름차순으로 유사한 노래 우선순
def make_dup(rating_idx,distances,indices,l2,song):
    dup=[] #중복 줄번호

    df_1=[]
    df_2=[]
    song_length=len(song)
    for j in range(song_length):
        if song[j] in rating_idx:
            idx=rating_idx.index(song[j])
            df_1.extend(indices[idx])
            df_2.extend(distances[idx])
    df1=pd.DataFrame(df_1,columns=['idx'])
    df2=pd.DataFrame(df_2,columns=['distance'])
    df=pd.concat([df1,df2],axis=1)
    if len(df)>0:
        df=df.sort_values(by='distance')
        dup=list(set(df[df.duplicated(['idx'])].idx))
        df=df.drop_duplicates('idx',keep='first') #중복 제거 최소값 keep
        df=df[df.idx.apply(lambda x: x in dup)] #중복 노래 거리
        dup=list(df.idx)
        return dup
    else:
        return 1   ## song에 있는 모든 노래가 추출한 노래에 없는 경
    
#노래 번호를 index로 변환
def ans_num(ans_song,rating_idx): 
    ans_song_num=[]
    song_l=len(ans_song[-1])
    for j in range(song_l):
        if ans_song[-1][j] in rating_idx:
            ans_song_num.append(rating_idx.index(ans_song[-1][j])) 
    return ans_song_num

#추천한 노래의 좋아요 합을 내림차순으로 태그 추천
def make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value):  
    tag_length=len(tag_col1)
    like_s=[]
    for j in ans_song_num:
        like_s.append(rating_value[j])
    like_sum1=list(pd.DataFrame(like_s).sum())
    if len(like_sum1)>0:
        
        for j in range(tag_length-1):   ##내림차순으로 선택 정렬
            for k in range(j+1,tag_length):
                if like_sum1[j]<like_sum1[k]:
                    like_sum1[j],like_sum1[k]=like_sum1[k],like_sum1[j]
                    tag_col1[j],tag_col1[k]=tag_col1[k],tag_col1[j]
                
        count_tag=len(ans_tag_1[-1])
        for j in range(tag_length):
            if tag_col1[j] not in ans_tag[-1] and count_tag<10:
                if like_sum1[j]==0:                            ##좋아요 합이 0이상인 태그만 넣도록
                    break
                else:
                    
                    ans_tag[-1].append(tag_col1[j])
                    ans_tag_1[-1].append(tag_col1[j])
                    count_tag+=1
                    if count_tag==10:
                        break
    return ans_tag,ans_tag_1

#추천한 노래중에 가장 큰 좋아요를 받은 태그 인덱스 추출
def make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value):
    rating_length=len(rating_value)
    tag_length=len(tag_col1)
    tag_idx=[]
    new_tag_idx=[]
    like=[]
    take_song=[]
    for j in range(len(ans_tag[-1])): 
        if ans_tag[-1][j] in tag_col1:
            
            tag_idx.append(tag_col1.index(ans_tag[-1][j]))
    
    
    for j in tag_idx:          #추출한 태그와 매핑된 노래
        for k in range(rating_length):
            if rating_value[k][j]>0:
                take_song.append(k)
    take_song=list(set(take_song))
    
    for j in take_song:  
        max1=0
        for k in range(tag_length):
            if rating_value[j][k]>max1 and k not in tag_idx and k not in new_tag_idx:
                max1=rating_value[j][k]
                new_tag=k
                
        if max1>0: 
            new_tag_idx.append(new_tag)
            like.append(max1)
        
    for j in range(len(new_tag_idx)-1):  ##좋아요 내림차순 정렬
        for k in range(j+1,len(new_tag_idx)):
            if like[j]<like[k]:
                like[j],like[k] = like[k],like[j]
                new_tag_idx[j],new_tag_idx[k] = new_tag_idx[k],new_tag_idx[j]
    
    count=len(ans_tag_1[-1])
    if count<10:  #태그 추
        for j in new_tag_idx:
            ans_tag[-1].append(tag_col1[j])
            ans_tag_1[-1].append(tag_col1[j])
            count+=1
            if count==10:
                break
    return ans_tag,ans_tag_1
        
        
        
##태그와 매핑된 노래에 대한 좋아요 합 내림차순으로 노래 추천 
def song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date):
    rating_length=len(rating_value)  
    count=len(ans_song_1[-1])
    tag_index=[]
    
    for j in ans_tag[-1]:
        if j in tag_col:              
            tag_index.append(tag_col.index(j))   
    

    tag_sum=[]
    for j in range(rating_length):   
        a=0
        for k in tag_index:
            a+=rating_value[j][k]
        tag_sum.append(a)
                
                
    tag_song=pd.DataFrame(tag_sum,index=rating_idx)
    tag_song=tag_song.sort_values(by=0,ascending=False)
    song_num=list(tag_song.index)
    
    p=0
    while count<100:  ##100개 될 때까지 채우기
        song_num_idx=rating_idx.index(song_num[p])                
        if song_num[p] not in ans_song[-1] and v_date>iss_date[song_num_idx]:
            ans_song[-1].append(song_num[p])
            ans_song_1[-1].append(song_num[p])
            count+=1
        p+=1
    return ans_song,ans_song_1

#Word2Vec 유사한 태그 5개 추천
def fill_5_tags(fill_tag,voca,word_tag):   
    tag_similar = []
    ans=[]
    for i in range(len(fill_tag)):
        if fill_tag[i] in voca:           
            model_result = model.wv.most_similar(fill_tag[i] , topn = 40)           
            for j in range(len(model_result)):                
                tag_similar.append(model_result[j])               
    tag_similar = sorted(tag_similar, key=lambda x: x[1], reverse=True)
        
    for k in range(len(tag_similar)):
        if tag_similar[k][0] not in ans and tag_similar[k][0] in word_tag:            
            ans.append(tag_similar[k][0])
            if len(ans)==5:
                break
        
    ans_tag = ans+fill_tag
    ans_tag_1 = ans

    return ans_tag_1, ans_tag


#태그와 매핑된 노래 좋아요 합으로 노래 50개 추
def tag_to_song(fill_tag,word_tag,rating_value,rating_idx,length_song,iss_date,v_date):


    like_cnt_dic = {} # 노래 번호 : 각 테그에 찍힌 좋아요의 총합을 연결해주는 딕셔너리
    like_cnt_spare_dic = {} #  like_cnt_dic이 0일때 대비해서 만드는 딕셔너리
    fill_tag_idx=[]
    for i in range(len(fill_tag)):
        if fill_tag[i] in word_tag:
            idx=word_tag.index(fill_tag[i])
            fill_tag_idx.append(idx)
        

    for i in range(length_song):
            
        song_tag = []
        
        for j in range(len(fill_tag_idx)):
            song_tag.append(rating_value[i][fill_tag_idx[j]])        
        
        if all(song_tag) > 0 and len(song_tag)>0 and iss_date[i]<=v_date:
            like_cnt_dic[sum(song_tag)] = rating_idx[i]
            
        elif any(song_tag) > 0 and len(song_tag)>0 and iss_date[i]<=v_date:
            like_cnt_spare_dic[sum(song_tag)] = rating_idx[i]
            



    if len(like_cnt_dic) > 50:

        ans_song = []                

        like_cnt_dic_keys = list(like_cnt_dic.keys())
        like_cnt_dic_keys.sort(reverse=True)
        like_cnt_dic_keys = like_cnt_dic_keys[:50]
            
        for i in range(50):
            ans_song.append(like_cnt_dic[like_cnt_dic_keys[i]])
                
    else:
    
        ans_song = []    
        
        like_cnt_dic_keys = list(like_cnt_dic.keys())
        like_cnt_dic_keys.sort(reverse=True)

        fill_like_cnt_dic_keys = len(like_cnt_dic_keys)
        lack_like_cnt_dic_keys = 50-len(like_cnt_dic_keys)
    
        like_cnt_spare_dic_keys = list(like_cnt_spare_dic.keys())
        like_cnt_spare_dic_keys.sort(reverse=True)
        like_cnt_spare_dic_keys = like_cnt_spare_dic_keys[:lack_like_cnt_dic_keys]
        
        for i in range(fill_like_cnt_dic_keys):
            ans_song.append(like_cnt_dic[like_cnt_dic_keys[i]])
            
        if len(like_cnt_spare_dic_keys) > lack_like_cnt_dic_keys:
        
            for j in range(lack_like_cnt_dic_keys):
                ans_song.append(like_cnt_spare_dic[like_cnt_spare_dic_keys[j]])
        else:
            for j in range(len(like_cnt_spare_dic_keys)):
                ans_song.append(like_cnt_spare_dic[like_cnt_spare_dic_keys[j]])

    return ans_song    

#플레이리스트 타이틀에서 장르 추출
def find_genre(title,gnr_code):
    korean = []
    english = []

    korean = re.compile('[가-힣]+').findall(str(title)) # 타이틀에서 한글만 추출
    english = [re.sub('[^a-zA-Z0-9]',' ',str(title)).strip().upper()] # 타이틀에서 영어만 추출

    genre_list = [] 


    for i in range(len(korean)):
        if korean[i] in list(gnr_code.gnr_name):
            genre_list.append((gnr_code[gnr_code.gnr_name.apply(lambda x: x == str(korean[i]))].gnr_code).values[0])
    for j in range(len(english)):
        if english[j] in list(gnr_code.gnr_name):
            genre_list.append((gnr_code[gnr_code.gnr_name.apply(lambda x: x == str(english[j]))].gnr_code).values[0])
    
    return genre_list

#추출된 장르에서 가장 많이 매핑된 노래
def genre_song(genre_list,rt_count4):
    search_song=rt_count4[rt_count4.song_gn_gnr_basket.apply(lambda x: set(x)==set(genre_list))]
    if len(search_song)>0:
        g_song=list(set(search_song.song.values))
        return g_song
    else:
        return -1
    
#플레이리스트 타이틀에서 장르 추출
def search_tag(word_tag,title_list):
    search=[]
    similar=[]
    cut=[]
    for i in title_list:
        if i in voca:
            model_result = model.wv.most_similar(i , topn = 40)
            cut.append(i)
            for j in range(len(model_result)):                
                similar.append(model_result[j])
            
                
    similar = sorted(similar, key=lambda x: x[1], reverse=True)
        
    if len(cut)+len(search)<5:
        
        for k in range(len(similar)):
            if similar[k][0] not in search and similar[k][0] in word_tag and similar[k][0] not in cut:            
                search.append(similar[k][0])

                if len(set(search+cut))>=5:
                    break
        
    tag_1 = list(set(search+cut))
    tag_2 = list(set(search+cut))

    return tag_1, tag_2

#플레이리스트 타이틀에서 아티스트 추출, 아티스트와 관련된 노래 추출
def artist_song(artist_list,token_list,v_date,song_meta,rating_idx):
    song=[]
    for i in token_list:
        if i in artist_list:
            ar_song=list(song_meta[(song_meta.artist_name_basket.apply(lambda x : i in x)) & (song_meta.issue_date<=v_date)].id.values)
            for j in ar_song:
                if len(song)==100:
                    break
                if j in rating_idx and j not in song:
                    song.append(j)
    return song
            

##한 번이라도 중복 된 것 중에서 최소 거리가 짧은 것 우선
##중복 노래 채우기 -> 태그 채우기(좋아요 많은 순 or 출현 빈도 수 => 태그로 나머지 노래 채우기)

ans_song=[]
ans_song_1=[]
ans_tag=[]
ans_tag_1=[]
tag_col=copy.deepcopy(word_tag)
l=len(test)
date_df=pd.DataFrame(test.updt_date)
date_df['updt_date']=pd.to_datetime(date_df['updt_date'])
val_date=copy.deepcopy(list(date_df.updt_date))
ss=list(map(int,song_count.index))
length_song=rating.shape[0]

for i in range(l):
    t=0
    song=copy.deepcopy(test.songs[i])  
    val_tag=copy.deepcopy(test.tags[i])
    v_date=val_date[i]
    l2=len(song)
    l4=len(val_tag)
    if l2!=0: #1,2번 경우
        ans_song.append(copy.deepcopy(song))
        ans_song_1.append([])
        dup=make_dup(rating_idx,distances,indices,l2,song) #중복노래 추출
        
        if dup==1: ## song에 있는 모든 노래가 추출된 노래에 없음
            if l4>0: 
                #유사한 태그 추출
                t=1
                tag_1,tag_2=fill_5_tags(val_tag,voca,word_tag)
                ans_tag.append(tag_2)
                ans_tag_1.append(tag_1)
                #추출한 태그를 통해 노래 추천
                a_song=tag_to_song(tag_2,word_tag,rating_value,rating_idx,length_song,iss_date,v_date)
                ans_song[-1].extend(copy.deepcopy(a_song))
                ans_song_1[-1].extend(copy.deepcopy(a_song))
                dup=make_dup(rating_idx,distances,indices,len(a_song),a_song) #추천한 노래의 중복 이웃 추출
                if dup!=1:
                    dup_l=len(dup)
                    count=len(ans_song_1[-1])
                    for j in dup: 
                        if rating_idx[j] not in ans_song[-1] and v_date>=iss_date[j]:
                            ans_song[-1].append(rating_idx[j])
                            ans_song_1[-1].append(rating_idx[j])
                            if len(ans_song_1[-1])==100:
                                break
                    
        
                count_tag=len(ans_tag_1[-1])        
                if count_tag<10:
            
                    ans_song_num=ans_num(ans_song,rating_idx)
                    tag_col1=copy.deepcopy(tag_col)         
                    ans_tag,ans_tag_1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
        
                if len(ans_song_1[-1])<100:
                    ans_song,ans_song_1=song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date)
                    
                if len(ans_tag_1[-1])<10:    
                    tag_col1=copy.deepcopy(tag_col)
                    ans_tag,ans_tag_1=make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value)
                
                #부족한 부분 추천 불가, 랜덤 추출
                while len(ans_song_1[-1])<100:
                    ans_song_1[-1].append(random.choice(ss))
                    ans_song_1[-1]=list(set(ans_song_1[-1]))
                while len(ans_tag_1[-1])<10:
                    ans_tag_1[-1].append(random.choice(word_tag))
                    ans_tag_1[-1]=list(set(ans_tag_1[-1]))
                
            else:
                #대체 노래 추출
                X=copy.deepcopy(ans_song[-1])
                ans=no_sub(X,test,song_meta,song_gr,song_c,v_date,rating_idx)
                if ans==0: #대체 노래 추출 불가
                    while len(ans_song_1[-1])<100:
                        ans_song_1[-1].append(random.choice(ss))
                        ans_song_1[-1]=list(set(ans_song_1[-1]))
                    while len(ans_tag_1[-1])<10:
                        ans_tag_1[-1].append(random.choice(word_tag))
                        ans_tag_1[-1]=list(set(ans_tag_1[-1]))
                else:
                    ans_song[-1].extend(ans)
                    ans_song_1[-1].extend(ans)
                    if len(ans)>1:
                        dup=make_dup(rating_idx,distances,indices,len(ans),ans) 
                        if dup!=1:
                            dup_l=len(dup)
                            count=len(ans_song_1[-1])
                            for j in dup:
                                if rating_idx[j] not in ans_song[-1] and count<100 and v_date>=iss_date[j]:
                                    ans_song[-1].append(rating_idx[j])
                                    ans_song_1[-1].append(rating_idx[j])
                                    count+=1
                                    if count==100:
                                        break
        
        else:
            dup_l=len(dup)
            if dup_l>1:
                count=len(ans_song_1[-1])
                for j in dup: 
                    if rating_idx[j] not in ans_song[-1] and count<100 and v_date>=iss_date[j]:
                        ans_song[-1].append(rating_idx[j])
                        ans_song_1[-1].append(rating_idx[j])
                        count+=1
                        if count==100:
                            break
                    
        
        if t!=1:
            tag=copy.deepcopy(list(test.tags[i]))
            l3=len(tag)
            if l3>0:
                ans_tag.append(copy.deepcopy(tag))
                ans_tag_1.append([])
            else:
                ans_tag.append([])
                ans_tag_1.append([])
            
            
            ans_song_num=ans_num(ans_song,rating_idx) 
            tag_col1=copy.deepcopy(tag_col) 
            ans_tag,ans_tag_1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
        count=len(ans_song_1[-1])
        if count<100:
            ans_song,ans_song_1=song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date)         
        count_tag=len(ans_tag_1[-1])
        
        if count_tag<10: 
            ans_song_num=ans_num(ans_song,rating_idx)
            tag_col1=copy.deepcopy(tag_col) 
            ans_tag,ans_tag1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)

        if len(ans_tag_1[-1])<10:    
            tag_col1=copy.deepcopy(tag_col)
            ans_tag,ans_tag_1=make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value)
            
        
    elif l4>0:  #3번 경우
        #유사한 태그 5개 추천
        tag_1,tag_2=fill_5_tags(val_tag,voca,word_tag)
        ans_tag.append(tag_2)
        ans_tag_1.append(tag_1)
    
        a_song=tag_to_song(tag_2,word_tag,rating_value,rating_idx,length_song,iss_date,v_date)
        ans_song.append(copy.deepcopy(a_song))
        ans_song_1.append(copy.deepcopy(a_song))
        dup=make_dup(rating_idx,distances,indices,len(a_song),a_song) 
        if dup!=1:
            dup_l=len(dup)
            count=len(ans_song_1[-1])
            for j in dup: 
                if rating_idx[j] not in ans_song[-1] and v_date>=iss_date[j]:
                    ans_song[-1].append(rating_idx[j])
                    ans_song_1[-1].append(rating_idx[j])
                    if len(ans_song_1[-1])==100:
                        break
                    
        count_tag=len(ans_tag_1[-1])        
        if count_tag<10:   
            
            ans_song_num=ans_num(ans_song,rating_idx)
        
            tag_col1=copy.deepcopy(tag_col) 
        
            ans_tag,ans_tag_1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
        
        if len(ans_song_1[-1])<100:
            ans_song,ans_song_1=song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date)
        
        if len(ans_tag_1[-1])<10:  
            tag_col1=copy.deepcopy(tag_col) 
            ans_tag,ans_tag_1=make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value)
            
        while len(ans_song_1[-1])<100:
            ans_song_1[-1].append(random.choice(ss))
            ans_song_1[-1]=list(set(ans_song_1[-1]))
        while len(ans_tag_1[-1])<10:
            ans_tag_1[-1].append(random.choice(word_tag))
            ans_tag_1[-1]=list(set(ans_tag_1[-1]))
        
        
    elif l2==0 and l4==0:  #4번 경우
        token_list=copy.deepcopy(ply_token_s.loc[i])
        tag_1,tag_2=search_tag(word_tag,token_list)
        ans_tag.append(tag_1)
        ans_tag_1.append(tag_2)
        if len(tag_1)!=0: 
            a_song=tag_to_song(tag_1,word_tag,rating_value,rating_idx,length_song,iss_date,v_date)
            ans_song.append(copy.deepcopy(a_song))

            ans_song_1.append(copy.deepcopy(a_song))
            dup=make_dup(rating_idx,distances,indices,len(a_song),a_song)
            if dup!=1:
                dup_l=len(dup)
                count=len(ans_song_1[-1])
                for j in dup: 
                    if rating_idx[j] not in ans_song[-1] and v_date>=iss_date[j]:
                        ans_song[-1].append(rating_idx[j])
                        ans_song_1[-1].append(rating_idx[j])
                        if len(ans_song_1[-1])==100:
                            break
                        
            count_tag=len(ans_tag_1[-1])        
            if count_tag<10:   
                
                ans_song_num=ans_num(ans_song,rating_idx)
                tag_col1=copy.deepcopy(tag_col) 
                ans_tag,ans_tag_1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
            
            if len(ans_song_1[-1])<100:
                ans_song,ans_song_1=song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date)
            
            if len(ans_tag_1[-1])<10: 
                tag_col1=copy.deepcopy(tag_col) 
                ans_tag,ans_tag_1=make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value)
            
            while len(ans_song_1[-1])<100:
                ans_song_1[-1].append(random.choice(ss))
                ans_song_1[-1]=list(set(ans_song_1[-1]))
            while len(ans_tag_1[-1])<10:
                ans_tag_1[-1].append(random.choice(word_tag))
                ans_tag_1[-1]=list(set(ans_tag_1[-1]))
        
        #플레이리스트 타이틀에서 장르 추출
        else: 
            genre_list=find_genre(token_list,gnr_code)
            if len(genre_list)>0:
                gr_song=genre_song(genre_list,rt_count4)
                if gr_song!=-1:
                    ans_song.append(copy.deepcopy(gr_song))
                    ans_song_1.append(copy.deepcopy(gr_song))
                    dup=make_dup(rating_idx,distances,indices,len(gr_song),gr_song) 
                    
                    if dup==1: 
                        if l4>0: 
                            t=1
                            tag_1,tag_2=fill_5_tags(val_tag,voca,word_tag)
                            ans_tag[-1].extend(tag_2)
                            ans_tag_1[-1].extend(tag_1)
                    
                            a_song=tag_to_song(tag_2,word_tag,rating_value,rating_idx,length_song,iss_date,v_date)
                            ans_song[-1].extend(copy.deepcopy(a_song))
                            ans_song_1[-1].extend(copy.deepcopy(a_song))
                            dup=make_dup(rating_idx,distances,indices,len(a_song),a_song)
                            if dup!=1:
                                dup_l=len(dup)
                                count=len(ans_song_1[-1])
                                for j in dup:
                                    if rating_idx[j] not in ans_song[-1] and v_date>=iss_date[j]:
                                        ans_song[-1].append(rating_idx[j])
                                        ans_song_1[-1].append(rating_idx[j])
                                        if len(ans_song_1[-1])==100:
                                            break
                                
                    
                            count_tag=len(ans_tag_1[-1])        
                            if count_tag<10: 
                                ans_song_num=ans_num(ans_song,rating_idx)
                                tag_col1=copy.deepcopy(tag_col) 
                                ans_tag,ans_tag_1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
                    
                            if len(ans_song_1[-1])<100:
                                ans_song,ans_song_1=song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date)
                            if len(ans_tag_1[-1])<10:  
                                tag_col1=copy.deepcopy(tag_col) 
                                ans_tag,ans_tag_1=make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value)
                    
                            while len(ans_song_1[-1])<100:
                                ans_song_1[-1].append(random.choice(ss))
                                ans_song_1[-1]=list(set(ans_song_1[-1]))
                            while len(ans_tag_1[-1])<10:
                                ans_tag_1[-1].append(random.choice(word_tag))
                                ans_tag_1[-1]=list(set(ans_tag_1[-1]))
                            
                        else:
                            
                            X=copy.deepcopy(ans_song[-1])
                            ans=no_sub(X,test,song_meta,song_gr,song_c,v_date,rating_idx)
                            if ans==0:
                                while len(ans_song_1[-1])<100:
                                    ans_song_1[-1].append(random.choice(ss))
                                    ans_song_1[-1]=list(set(ans_song_1[-1]))
                                while len(ans_tag_1[-1])<10:
                                    ans_tag_1[-1].append(random.choice(word_tag))
                                    ans_tag_1[-1]=list(set(ans_tag_1[-1]))
                            else:
                                ans_song[-1].extend(ans)
                                ans_song_1[-1].extend(ans)
                                if len(ans)>1:
                                    dup=make_dup(rating_idx,distances,indices,len(ans),ans)
                                    if dup!=1:
                                        dup_l=len(dup)
                                        count=len(ans_song_1[-1])
                                        for j in dup: 
                                            if rating_idx[j] not in ans_song[-1] and count<100 and v_date>=iss_date[j]:
                                                ans_song[-1].append(rating_idx[j])
                                                ans_song_1[-1].append(rating_idx[j])
                                                count+=1
                                                if count==100:
                                                    break
                    
                    else:
            
                        dup_l=len(dup)
                    
                        if dup_l>1:                
                            count=len(ans_song_1[-1])
                            for j in dup:
                                if rating_idx[j] not in ans_song[-1] and count<100 and v_date>=iss_date[j]:
                                    ans_song[-1].append(rating_idx[j])
                                    ans_song_1[-1].append(rating_idx[j])
                                    count+=1
                                    if count==100:
                                        break
                                
                    if t!=1:
                        
                        
                        ans_song_num=ans_num(ans_song,rating_idx) 
                        
                        
                        tag_col1=copy.deepcopy(tag_col)
                        
                        ans_tag,ans_tag_1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
                        
                    
                    count=len(ans_song_1[-1])
                    
                    if count<100:
                        ans_song,ans_song_1=song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date) 
                    
            
                    
                    count_tag=len(ans_tag_1[-1])
                    if count_tag<10: 
                        
                        ans_song_num=ans_num(ans_song,rating_idx)
                    
                        tag_col1=copy.deepcopy(tag_col)
                    
                        ans_tag,ans_tag1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
                    
                    
                    if len(ans_tag_1[-1])<10:  
                        tag_col1=copy.deepcopy(tag_col)
                        ans_tag,ans_tag_1=make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value)
                    
                    
            #플레이리스트 타이틀에서 아티스트 명 추출
            else:
                art_song=artist_song(artist_list,token_list,v_date,song_meta,rating_idx)
                if len(art_song)>0:
                    ans_song.append(copy.deepcopy(art_song))
                    ans_song_1.append(copy.deepcopy(art_song))
                    
                    dup=make_dup(rating_idx,distances,indices,len(art_song),art_song) 
                    
                    if dup==1: 
                        if l4>0: 
                            t=1
                            tag_1,tag_2=fill_5_tags(val_tag,voca,word_tag)
                            ans_tag[-1].extend(tag_2)
                            ans_tag_1[-1].extend(tag_1)
                    
                            #노래 50개 채우기
                            a_song=tag_to_song(tag_2,word_tag,rating_value,rating_idx,length_song,iss_date,v_date)
                            ans_song[-1].extend(copy.deepcopy(a_song))
            
                            ans_song_1[-1].extend(copy.deepcopy(a_song))
                            dup=make_dup(rating_idx,distances,indices,len(a_song),a_song)
                            if dup!=1:
                                dup_l=len(dup)
                                count=len(ans_song_1[-1])
                                for j in dup:
                                    if rating_idx[j] not in ans_song[-1] and v_date>=iss_date[j]:
                                        ans_song[-1].append(rating_idx[j])
                                        ans_song_1[-1].append(rating_idx[j])
                                        if len(ans_song_1[-1])==100:
                                            break
                                
                    
                            count_tag=len(ans_tag_1[-1])        
                            if count_tag<10:
                        
                                ans_song_num=ans_num(ans_song,rating_idx)
                                tag_col1=copy.deepcopy(tag_col) 
                                ans_tag,ans_tag_1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
                    
                            if len(ans_song_1[-1])<100:
                                ans_song,ans_song_1=song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date)
                    
                        
                            if len(ans_tag_1[-1])<10:   
                                tag_col1=copy.deepcopy(tag_col)
                                ans_tag,ans_tag_1=make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value)
                    
                            while len(ans_song_1[-1])<100:
                                ans_song_1[-1].append(random.choice(ss))
                                ans_song_1[-1]=list(set(ans_song_1[-1]))
                            while len(ans_tag_1[-1])<10:
                                ans_tag_1[-1].append(random.choice(word_tag))
                                ans_tag_1[-1]=list(set(ans_tag_1[-1]))
                            
                        else:
                            
                            X=copy.deepcopy(ans_song[-1])
                            ans=no_sub(X,test,song_meta,song_gr,song_c,v_date,rating_idx)
                            if ans==0: 
                                while len(ans_song_1[-1])<100:
                                    ans_song_1[-1].append(random.choice(ss))
                                    ans_song_1[-1]=list(set(ans_song_1[-1]))
                                while len(ans_tag_1[-1])<10:
                                    ans_tag_1[-1].append(random.choice(word_tag))
                                    ans_tag_1[-1]=list(set(ans_tag_1[-1]))
                            else:
                                ans_song[-1].extend(ans)
                                ans_song_1[-1].extend(ans)
                                if len(ans)>1:
                                    dup=make_dup(rating_idx,distances,indices,len(ans),ans) 
                                    if dup!=1:
                                        dup_l=len(dup)
                                        count=len(ans_song_1[-1])
                                        for j in dup: 
                                            if rating_idx[j] not in ans_song[-1] and count<100 and v_date>=iss_date[j]:
                                                ans_song[-1].append(rating_idx[j])
                                                ans_song_1[-1].append(rating_idx[j])
                                                count+=1
                                                if count==100:
                                                    break
                    
                    else:
                        dup_l=len(dup)
                    
                        if dup_l>1:
                            count=len(ans_song_1[-1])
                            for j in dup: 
                                if rating_idx[j] not in ans_song[-1] and count<100 and v_date>=iss_date[j]:
                                    ans_song[-1].append(rating_idx[j])
                                    ans_song_1[-1].append(rating_idx[j])
                                    count+=1
                                    if count==100:
                                        break
                    if t!=1:
                    
                        ans_song_num=ans_num(ans_song,rating_idx) 
                        
                        
                        tag_col1=copy.deepcopy(tag_col) 
                        
                        ans_tag,ans_tag_1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
                        
                    
                    count=len(ans_song_1[-1])
                    
                    
                    if count<100:
                        ans_song,ans_song_1=song_else(ans_song_1,ans_song,ans_tag,tag_col,rating_value,rating_idx,v_date,iss_date) 
                    
                    count_tag=len(ans_tag_1[-1])
                    if count_tag<10: 
                        
                        ans_song_num=ans_num(ans_song,rating_idx)
                    
                        tag_col1=copy.deepcopy(tag_col) 
                    
                        ans_tag,ans_tag1=make_tag(ans_tag_1,ans_tag,ans_song_num,tag_col1,rating_value)
                    
                    
                    if len(ans_tag_1[-1])<10:   
                        tag_col1=copy.deepcopy(tag_col) 
                        ans_tag,ans_tag_1=make_tag2(ans_tag_1,ans_tag,tag_col1,rating_value)                       
                    
                else:
                    ans_song_1.append([])
                    ans_song.append([])
                    while len(ans_song_1[-1])<100:
                        ans_song_1[-1].append(random.choice(ss))
                        ans_song_1[-1]=list(set(ans_song_1[-1]))
                        ans_song[-1].append(random.choice(ss))
                        ans_song[-1]=list(set(ans_song[-1]))
                    while len(ans_tag_1[-1])<10:
                        ans_tag_1[-1].append(random.choice(word_tag))
                        ans_tag_1[-1]=list(set(ans_tag_1[-1]))
                        ans_tag[-1].append(random.choice(word_tag))
                        ans_tag[-1]=list(set(ans_tag[-1]))
                    
    while len(ans_song_1[-1])<100:
        ans_song_1[-1].append(random.choice(ss))
        ans_song_1[-1]=list(set(ans_song_1[-1]))
    while len(ans_tag_1[-1])<10:
        ans_tag_1[-1].append(random.choice(word_tag))
        ans_tag_1[-1]=list(set(ans_tag_1[-1]))
      

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
id_list=list(test.id)
answer=[]
for i in range(len(test)):
    dic={'id':id_list[i], 'songs':ans_song_1[i], 'tags':ans_tag_1[i]}
    answer.append(dic)
with open('results.json', 'w', encoding='UTF-8') as file:
    file.write(json.dumps(answer,cls=NpEncoder, ensure_ascii=False,indent='\t'))

