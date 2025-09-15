https://ithelp.ithome.com.tw/m/articles/10219033

https://jasmine880809.medium.com/recommender-system-%E6%8E%A8%E8%96%A6%E7%B3%BB%E7%B5%B1-content-based-filtering-and-collaborative-filtering-9d338b7b22bd

https://www.youtube.com/watch?v=cvR0FaI486k&pp=ygUc5o6o6Jam566X5rOVIOaFouaYr-ayiOaAnemMhA%3D%3D

https://jasmine880809.medium.com/python-netflix-%E6%95%B8%E6%93%9A%E5%88%86%E6%9E%90-%E5%BD%B1%E7%89%87%E6%8E%A8%E8%96%A6%E6%BC%94%E7%AE%97%E6%B3%95-%E9%99%84%E5%AE%8C%E6%95%B4%E7%A8%8B%E5%BC%8F%E7%A2%BC-753bbc1c6212

application
* drug repurposing
* music recommend

method
* content based filtering: 
    以內容為基礎，比較商品屬性，找到最相似的商品。
    - moodify的內容
        如果是文字資料，像是描述電影的文字，進行text to vector的轉換，要one-hot encoding 或是 word2vec。
        然後最後計算cosine similarity。e.g.  Kaggle 的 TMDB 5000 Movie Dataset

* collaborative filtering: 
    集合眾人意見，找出最相似的顧客或是找出最相似的商品，進而進行推薦

    - User-based: 與你相似的用戶也購買了。
    - item-based: 購買此商品的人也買了。

    - memery-based
    - model-based
        - 以過去的碩士班論文研究，製作過比較偏向model-based的內容。製作大型的knowledge graph 然後進行matrix factorization (SVD)。
            最後也是使用cosine similarity 計算基因與藥物的距離。集合起來，可以計算pathway vs disease, pathway vs drug, and drug vs disease. 
            繼續往後，可以嘗試reinforcement learning? GNN, GCN 還有大型語言模型。

* hybrid aporoach

* code implement
    - similarity
        1. jaccard similarity
        2. cosine similarity
        3. pearson 
    - matrix factorization
        factor 數量的決定。 rating matrix(user x item) = user matrix (user x factor) x item matrix(factor x item)
        objective function
    - KNN
