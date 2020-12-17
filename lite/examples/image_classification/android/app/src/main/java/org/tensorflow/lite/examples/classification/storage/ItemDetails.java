package org.tensorflow.lite.examples.classification.storage;

public class ItemDetails {
    private String Id;
    private String price;
    private String imageUrl;

    public ItemDetails(String id, String price, String imageUrl) {
        Id = id;
        this.price = price;
        this.imageUrl = imageUrl;
    }

    public ItemDetails() {
    }

    public String getId() {
        return Id;
    }

    public void setId(String id) {
        Id = id;
    }

    public String getPrice() {
        return price;
    }

    public void setPrice(String price) {
        this.price = price;
    }

    public String getImageUrl() {
        return imageUrl;
    }

    public void setImageUrl(String imageUrl) {
        this.imageUrl = imageUrl;
    }
}
